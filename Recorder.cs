using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text.Json;
using System.Text.Json.Nodes;
using HomefrontSDK;

namespace HomefrontAI;

/// <summary>HomefrontAI: records how real players move and act, ten times a second, into one file per map
/// (Mods\HomefrontAI\recordings\&lt;map&gt;_&lt;time&gt;.hfrec). train\hfimitate.py turns the recordings into
/// a policy and per-map heat maps for HomefrontBots. Chat: "!rec ...".</summary>
[Script(Name = "HomefrontAI", Description = "records real players' movement to train HomefrontBots (!rec in chat)")]
public sealed class Recorder : Script
{
    /// <summary>imitation.json in the mod's folder (written with the defaults when missing).</summary>
    sealed class Config
    {
        public bool Enabled { get; set; } = true;
        public bool RecordBots { get; set; } = true;     // bots too (flagged), to compare them with the humans
        public int MinHumans { get; set; } = 1;          // record only while at least this many humans play
        public bool Announce { get; set; } = true;       // tell joining players that movement is recorded
        public int MaxRecordingMB { get; set; } = 4096;  // stop recording once the folder holds this much
        public int MaxSoldiers { get; set; } = 64;       // per sample
    }

    static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    Config cfg = new();
    string recordingsDir = "";
    RecordingWriter? rec;
    long existingBytes;
    float clock, sinceFlush;
    bool capped;

    readonly Senses senses = new();
    readonly Dictionary<long, ushort> pids = new();     // controller + player id -> this recording's id
    readonly HashSet<nint> firing = new();              // pawns holding a trigger (Weapon.ServerStartFire..Stop)
    readonly HashSet<nint> jumped = new();              // pawns that jumped since the last sample
    readonly List<(Pawn Pawn, int Team)> soldiers = new();
    readonly float[] features = new float[Features.Count];

    string ConfigPath => Path.Combine(Directory, "imitation.json");

    public override void OnStart()
    {
        recordingsDir = Path.Combine(Directory, "recordings");
        System.IO.Directory.CreateDirectory(recordingsDir);
        LoadConfig();
        Bridge.Handle("imitation.status", _ => Status());
        // If a map is already running, the SDK calls OnMapLoaded next.
        Log.Write($"HomefrontAI started; recordings in {recordingsDir}; schema {Features.Schema}");
    }

    void LoadConfig()
    {
        try
        {
            if (File.Exists(ConfigPath))
                cfg = JsonSerializer.Deserialize<Config>(File.ReadAllText(ConfigPath), Json) ?? new Config();
            else
                File.WriteAllText(ConfigPath, JsonSerializer.Serialize(cfg, Json));
        }
        catch (Exception e) when (e is IOException or JsonException or UnauthorizedAccessException)
        {
            Log.Warn($"config {ConfigPath}: {e.Message}; using the defaults");
            cfg = new Config();
        }
    }

    void SaveConfig()
    {
        try
        {
            File.WriteAllText(ConfigPath, JsonSerializer.Serialize(cfg, Json));
        }
        catch (IOException e)
        {
            Log.Warn($"can't save {ConfigPath}: {e.Message}");
        }
    }

    public override void OnStop() => Close("mod stopping");

    public override void OnMapLoaded(GameInfo game)
    {
        Close("new map");
        if (cfg.Enabled && game is not FL2GameIntermission)
            Open(game);
    }

    public override void OnMapUnloading(string map) => Close("map unloading");

    public override void OnPlayerJoined(PlayerController player)
    {
        if (cfg.Enabled && cfg.Announce)
            player.Tell("[rec] this server records anonymous movement data (no names) to train its bots");
    }

    // ---- recording files

    static long FolderBytes(string dir)
    {
        long total = 0;
        try
        {
            foreach (var f in new DirectoryInfo(dir).EnumerateFiles("*.hfrec"))
                total += f.Length;
        }
        catch (IOException)
        {
        }
        return total;
    }

    void Open(GameInfo game)
    {
        existingBytes = FolderBytes(recordingsDir);
        capped = existingBytes >= cfg.MaxRecordingMB * 1024L * 1024L;
        if (capped)
        {
            Log.Warn($"recordings folder holds {existingBytes >> 20} MB (cap {cfg.MaxRecordingMB} MB): not recording; train and clear it");
            return;
        }
        string map = Game.MapName;
        string safe = string.Join("_", map.Split(Path.GetInvalidFileNameChars()));
        var now = DateTimeOffset.UtcNow;
        string path = Path.Combine(recordingsDir, $"{safe}_{now:yyyyMMdd_HHmmss}.hfrec");
        for (int n = 2; File.Exists(path); ++n)
            path = Path.Combine(recordingsDir, $"{safe}_{now:yyyyMMdd_HHmmss}_{n}.hfrec");
        try
        {
            rec = new RecordingWriter(path, map, game.ClassName, now);
        }
        catch (IOException e)
        {
            Log.Warn($"can't create {path}: {e.Message}");
            rec = null;
            return;
        }
        clock = sinceFlush = 0;
        pids.Clear();
        nextSample.Clear();
        firing.Clear();
        jumped.Clear();
        senses.Clear();
        Log.Write($"recording {map} ({game.ClassName}) to {Path.GetFileName(path)}");
    }

    void Close(string why)
    {
        if (rec == null)
            return;
        var r = rec;
        rec = null;
        try
        {
            r.Dispose();
        }
        catch (IOException)
        {
        }
        Log.Write($"closed {Path.GetFileName(r.Path)} ({why}): {r.SampleCount} samples, {r.KillCount} kills, {r.Bytes / 1024} KB");
        // A file with nothing in it (no humans played) isn't worth keeping.
        if (r.SampleCount == 0)
            try { File.Delete(r.Path); } catch (IOException) { }
    }

    // A controller's pointer can be reused by a later player: key on it and the player's id together.
    static long Key(Controller c) => ((long)(uint)c.Ptr << 32) | (uint)(c.PlayerReplicationInfo?.PlayerID ?? 0);

    ushort Pid(Controller c, int team, bool bot)
    {
        long key = Key(c);
        if (pids.TryGetValue(key, out ushort id))
            return id;
        id = (ushort)Math.Min(pids.Count, RecordingWriter.NoPid - 1);
        pids[key] = id;
        rec?.Soldier(id, (sbyte)team, bot);
        return id;
    }

    // A failed write (disk full) stops this recording, not the mod.
    void Failed(IOException e)
    {
        Log.Warn($"recording stopped: {e.Message}");
        try { rec?.Dispose(); } catch (IOException) { }
        rec = null;
    }

    // ---- sampling

    // Every soldier is sampled every SampleSeconds, but not all in the same tick: each has its own next time, so
    // the sight lines and probes (up to 12 traces a soldier) spread over the ticks in between.
    readonly Dictionary<long, float> nextSample = new();

    public override void OnTick(float deltaSeconds)
    {
        if (rec == null || !Game.MapLoaded)
            return;
        clock += deltaSeconds;
        sinceFlush += deltaSeconds;

        if (Game.Info is not GameInfo info || Game.World is not World world)
            return;
        if (info.bWaitingToStartMatch || info.bGameEnded)
            return;  // the lobby countdown and the end screen aren't play

        int humans = 0;
        soldiers.Clear();
        for (Pawn? p = Game.WorldInfo?.PawnList; p != null && soldiers.Count < 256; p = p.NextPawn)
        {
            if (!p.Alive || p.Health <= 0 || p.PlayerReplicationInfo?.Team is not TeamInfo t)
                continue;
            soldiers.Add((p, (int)t.TeamIndex));
            if (p.Controller is PlayerController)
                ++humans;
        }
        if (humans < cfg.MinHumans)
            return;

        try
        {
            int written = 0;
            foreach (var (pawn, team) in soldiers)
            {
                if (written >= cfg.MaxSoldiers || rec == null)
                    break;
                if (pawn.Controller is not Controller c || c.PlayerReplicationInfo is not PlayerReplicationInfo pri)
                    continue;
                bool bot = pri.bBot || c is not PlayerController;
                if (bot && !cfg.RecordBots)
                    continue;
                long key = Key(c);
                if (!nextSample.TryGetValue(key, out float due))
                    due = clock + Features.SampleSeconds * (nextSample.Count % 8) / 8f;   // stagger newcomers
                if (clock < due)
                {
                    nextSample[key] = due;
                    continue;
                }
                // Next due one interval on; a long hitch doesn't make a burst of catch-up samples.
                nextSample[key] = clock - due > Features.SampleSeconds ? clock + Features.SampleSeconds : due + Features.SampleSeconds;
                Sample(pawn, c, team, bot, world, info);
                jumped.Remove(pawn.Ptr);
                ++written;
            }

            if (sinceFlush >= 5f && rec != null)
            {
                sinceFlush = 0;
                rec.Flush();
                if (existingBytes + rec.Bytes >= cfg.MaxRecordingMB * 1024L * 1024L)
                {
                    Log.Warn($"recording cap of {cfg.MaxRecordingMB} MB reached");
                    capped = true;
                    Close("cap reached");
                }
            }
        }
        catch (IOException e)
        {
            Failed(e);
        }
    }

    void Sample(Pawn pawn, Controller c, int team, bool bot, World world, GameInfo info)
    {
        var r = rec!;
        ushort pid = Pid(c, team, bot);
        Pawn? preferred = c.Enemy is { IsValid: true } e && e.Alive ? e : null;
        Pawn? enemy = senses.Observe(pawn, team, soldiers, preferred, world, info, clock, out var ctx);
        Features.Write(ctx, features);

        var flags = (SampleFlags)0;
        if (bot) flags |= SampleFlags.Bot;
        if (pawn.bIsCrouched) flags |= SampleFlags.Crouched;
        if (pawn.bIsSprinting) flags |= SampleFlags.Sprinting;
        if (pawn.bIsWalking) flags |= SampleFlags.Walking;
        if (pawn.Physics == EPhysics.PHYS_Falling) flags |= SampleFlags.Falling;
        if (pawn.Physics == EPhysics.PHYS_Ladder) flags |= SampleFlags.Ladder;
        if (firing.Contains(pawn.Ptr)) flags |= SampleFlags.Firing;
        if (jumped.Contains(pawn.Ptr)) flags |= SampleFlags.Jumped;
        if (enemy != null) flags |= SampleFlags.HasEnemy;

        ushort enemyPid = RecordingWriter.NoPid;
        if (enemy?.Controller is Controller ec)
            enemyPid = Pid(ec, ec.TeamIndex, ec.PlayerReplicationInfo?.bBot == true || ec is not PlayerController);

        Vector3 accel = Senses.V(pawn.Acceleration);
        float rate = pawn.AccelRate > 1f ? pawn.AccelRate : 2048f;
        var sample = new Sample
        {
            Time = clock,
            Pid = pid,
            Team = (sbyte)team,
            Flags = flags,
            Health = Senses.HealthOf(pawn),
            Location = Senses.V(pawn.Location),
            Velocity = Senses.V(pawn.Velocity),
            Acceleration = accel,
            AccelRate = rate,
            GroundSpeed = pawn.GroundSpeed,
            Yaw = Senses.YawOf(pawn),
            Pitch = Senses.PitchOf(pawn),
            FloorZ = pawn.Location.Z - Senses.HalfHeight(pawn),
            Frame = Features.FrameOf(ctx),
            EnemyPid = enemyPid,
            Weapon = r.Intern(Senses.WeaponName(pawn)),
            Features = features,
        };
        r.Write(sample);
        // What this human did, for its next sample's prev_fwd/prev_right (a bot's own brain sets its own).
        senses.SetMove(pawn, new Vector2(accel.X, accel.Y) / rate);
    }

    // ---- events

    [Hook("Weapon", "ServerStartFire")]
    bool OnStartFire(Weapon weapon, HookFrame frame)
    {
        if (weapon.Instigator is Pawn p)
            firing.Add(p.Ptr);
        return true;
    }

    [Hook("Weapon", "ServerStopFire")]
    bool OnStopFire(Weapon weapon, HookFrame frame)
    {
        if (weapon.Instigator is Pawn p)
            firing.Remove(p.Ptr);
        return true;
    }

    // The return value of a script-to-script call goes to the caller, not the frame: whether the jump happened is
    // read off the pawn instead (DoJump puts it in the air, going up).
    [Hook("Pawn", "DoJump", After = true)]
    void OnJump(Pawn pawn, HookFrame frame)
    {
        if (jumped.Count > 512)
            jumped.Clear();   // pawns that are never sampled (bots, with recordBots off)
        if (pawn.Physics == EPhysics.PHYS_Falling && pawn.Velocity.Z > 50f)
            jumped.Add(pawn.Ptr);
    }

    [Hook("GameInfo", "Killed", After = true)]
    void OnKilled(GameInfo game, HookFrame frame)
    {
        if (rec == null)
            return;
        Controller? killer = frame.Object<Controller>("Killer"), killed = frame.Object<Controller>("Killed");
        Pawn? victim = frame.Object<Pawn>("KilledPawn");
        if (killed == null || victim == null)
            return;
        firing.Remove(victim.Ptr);
        ushort Id(Controller c) => Pid(c, c.TeamIndex, c.PlayerReplicationInfo?.bBot == true || c is not PlayerController);
        try
        {
            ushort killerId = killer != null && killer.Ptr != killed.Ptr ? Id(killer) : RecordingWriter.NoPid;
            Vector3 killerAt = killer?.Pawn is Pawn kp && kp.Alive ? Senses.V(kp.Location) : default;
            string damage = frame.Object<UClass>("DamageType")?.ObjectName ?? "";
            rec.Kill(clock, killerId, Id(killed), killerAt, Senses.V(victim.Location), rec.Intern(damage));
        }
        catch (IOException e)
        {
            Failed(e);
        }
    }

    // ---- commands and bridge

    JsonObject Status() => new()
    {
        ["enabled"] = cfg.Enabled,
        ["recordBots"] = cfg.RecordBots,
        ["capped"] = capped,
        ["file"] = rec == null ? null : Path.GetFileName(rec.Path),
        ["samples"] = rec?.SampleCount ?? 0,
        ["kills"] = rec?.KillCount ?? 0,
        ["bytes"] = rec?.Bytes ?? 0,
        ["folderBytes"] = existingBytes + (rec?.Bytes ?? 0),
        ["traces"] = senses.Traces,
        ["schema"] = Features.Schema,
    };

    const string Usage = "status | on | off | bots on|off | split";

    [Command("rec", Help = "!rec " + Usage)]
    void RecCommand(CommandContext ctx)
    {
        string what = ctx.Arg(0, "status");
        bool admin = ctx.Player == null || ctx.Player.PlayerReplicationInfo?.bAdmin == true;
        if (what != "status" && !admin)
        {
            ctx.Reply("[rec] only admins (or the bridge) can change recording");
            return;
        }
        switch (what)
        {
            case "status":
                ctx.Reply(rec == null
                    ? $"[rec] not recording ({(cfg.Enabled ? capped ? "folder full" : "between maps" : "off")})"
                    : $"[rec] {Path.GetFileName(rec.Path)}: {rec.SampleCount} samples, {rec.KillCount} kills, {rec.Bytes >> 10} KB" +
                      (cfg.RecordBots ? ", bots too" : ""));
                break;
            case "on":
                cfg.Enabled = true;
                SaveConfig();
                if (rec == null && Game.MapLoaded && Game.Info is GameInfo g && g is not FL2GameIntermission)
                    Open(g);
                ctx.Reply("[rec] recording on");
                break;
            case "off":
                cfg.Enabled = false;
                SaveConfig();
                Close("turned off");
                ctx.Reply("[rec] recording off");
                break;
            case "bots":
                cfg.RecordBots = ctx.Arg(1) != "off";
                SaveConfig();
                ctx.Reply($"[rec] bots {(cfg.RecordBots ? "recorded too" : "not recorded")}");
                break;
            case "split":
                if (Game.Info is GameInfo gi && cfg.Enabled)
                {
                    Close("split");
                    Open(gi);
                }
                ctx.Reply("[rec] " + (rec == null ? "not recording" : "new file " + Path.GetFileName(rec.Path)));
                break;
            default:
                ctx.Reply("[rec] " + Usage);
                break;
        }
    }
}
