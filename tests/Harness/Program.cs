using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text.Json;
using HomefrontAI;

// Harness synth <dir> [files]          scripted "players" on an open arena -> .hfrec files (real writer + features)
// Harness check <policy.json> <parity.json>   C# forward pass == trainer's; the policy learned the script
return args.Length >= 2 && args[0] == "synth" ? Synth.Run(args[1], args.Length > 2 ? int.Parse(args[2]) : 5)
     : args.Length >= 3 && args[0] == "check" ? Check.Run(args[1], args[2])
     : Usage();

static int Usage()
{
    Console.Error.WriteLine("Harness synth <dir> [files] | check <policy.json> <parity.json>");
    return 2;
}

/// <summary>Scripted soldiers. The humans' script, which the policy should learn: no enemy in sight -> run forward,
/// sprinting; enemy under 1500 -> strafe (right 65%, left 25%, still 10%, held 0.6-1.2 s, the other way when a
/// wall is in the way), the odd jump; 1500-2500 -> diagonal forward-right; beyond -> crouch and hold still.
/// Two of the soldiers are bots that wander at random (the trainer must leave them out).</summary>
static class Synth
{
    const float Arena = 9000f, Speed = 440f, Rate = 2048f, Dt = Features.SampleSeconds;

    sealed class Soldier
    {
        public ushort Pid;
        public sbyte Team;
        public bool Bot;
        public Vector3 Loc, Vel;
        public float Yaw, Health = 1f;
        public int Choice;
        public float HoldUntil;
        public Vector2 Prev;
        public float EngagedSince = -1;
        public bool Crouched, Sprinting, Jumped;
        public Vector2 Waypoint;
    }

    static float Probe(Vector3 at, Vector2 dir)
    {
        float best = Features.ProbeLength;
        if (dir.X > 1e-4f) best = MathF.Min(best, (Arena - at.X) / dir.X);
        if (dir.X < -1e-4f) best = MathF.Min(best, -at.X / dir.X);
        if (dir.Y > 1e-4f) best = MathF.Min(best, (Arena - at.Y) / dir.Y);
        if (dir.Y < -1e-4f) best = MathF.Min(best, -at.Y / dir.Y);
        return Math.Clamp(best / Features.ProbeLength, 0f, 1f);
    }

    public static int Run(string dir, int files)
    {
        Directory.CreateDirectory(dir);
        var rng = new Random(7);
        long total = 0;
        for (int file = 0; file < files; ++file)
        {
            string path = Path.Combine(dir, $"synth-arena_{file:00}.hfrec");
            if (File.Exists(path))
                File.Delete(path);
            using var w = new RecordingWriter(path, "synth-arena", "FL2GameTDM", DateTimeOffset.UtcNow.AddMinutes(file));
            var soldiers = new List<Soldier>();
            for (int i = 0; i < 8; ++i)
            {
                var s = new Soldier { Pid = (ushort)i, Team = (sbyte)(i % 2), Bot = i >= 6 };
                Respawn(s, rng);
                soldiers.Add(s);
                w.Soldier(s.Pid, s.Team, s.Bot);
            }
            ushort weapon = w.Intern(file % 2 == 0 ? "CFG_RIF_M4" : "CFG_SMG_DIABLO");
            var features = new float[Features.Count];
            float t = 0;
            for (int step = 0; step < 6000; ++step, t += Dt)   // 10 minutes
            {
                foreach (var s in soldiers)
                {
                    // Nearest enemy in sight (an open arena: range is sight).
                    Soldier? enemy = null;
                    float best = Features.SightRange;
                    int visible = 0, allies = 0, near = 0;
                    foreach (var o in soldiers)
                    {
                        if (o == s) continue;
                        float d = Vector3.Distance(o.Loc, s.Loc);
                        if (o.Team == s.Team) { if (d < Features.AllyRange) ++allies; continue; }
                        if (d < Features.NearEnemyRange) ++near;
                        if (d < Features.SightRange) { ++visible; if (d < best) { best = d; enemy = o; } }
                    }
                    if (enemy == null) s.EngagedSince = -1; else if (s.EngagedSince < 0) s.EngagedSince = t;
                    if (enemy == null && !s.Bot)
                    {
                        // View swings toward a wandering waypoint; it runs where it looks.
                        if (Vector2.Distance(s.Waypoint, new Vector2(s.Loc.X, s.Loc.Y)) < 300f)
                            s.Waypoint = new Vector2(500 + rng.NextSingle() * (Arena - 1000), 500 + rng.NextSingle() * (Arena - 1000));
                        var to = s.Waypoint - new Vector2(s.Loc.X, s.Loc.Y);
                        if (to.LengthSquared() > 1f)
                            s.Yaw = MathF.Atan2(to.Y, to.X);
                    }
                    var ctx = new Features.Context
                    {
                        Self = new Features.Soldier { Location = s.Loc, Velocity = s.Vel, Yaw = s.Yaw, Health = s.Health },
                        HasEnemy = enemy != null,
                        GroundSpeed = Speed,
                        VisibleEnemies = Math.Min(visible, 4), NearAllies = allies, NearEnemies = near,
                        EngagedSeconds = s.EngagedSince < 0 ? 0 : t - s.EngagedSince,
                        PreviousMove = s.Prev,
                        Crouched = s.Crouched, Sprinting = s.Sprinting, Falling = false,
                        WeaponGroup = Features.WeaponGroupOf(file % 2 == 0 ? "CFG_RIF_M4" : "CFG_SMG_DIABLO"),
                    };
                    if (enemy != null)
                        ctx.Enemy = new Features.Soldier { Location = enemy.Loc, Velocity = enemy.Vel, Yaw = enemy.Yaw, Health = enemy.Health };
                    Vector2 frame = Features.FrameOf(ctx);
                    ctx.Probe = new float[Features.Probes];
                    for (int i = 0; i < Features.Probes; ++i)
                        ctx.Probe[i] = Probe(s.Loc, Features.FromFrame(Features.Direction(i), frame));
                    Features.Write(ctx, features);

                    // The script decides this sample's input; crouch/sprint take effect from the next sample.
                    bool jumpedNow = false;
                    int cls;
                    bool crouch = false, sprint = false;
                    if (s.Bot)
                    {
                        if (t >= s.HoldUntil) { s.Choice = rng.Next(9); s.HoldUntil = t + 0.5f; }
                        cls = s.Choice;
                        crouch = rng.NextDouble() < 0.5;
                    }
                    else if (enemy == null)
                    {
                        cls = 1;
                        sprint = true;
                    }
                    else if (best < 1500f)
                    {
                        if (t >= s.HoldUntil || s.Choice is not (0 or 3 or 7))
                        {
                            double r = rng.NextDouble();
                            s.Choice = r < 0.65 ? 3 : r < 0.90 ? 7 : 0;
                            s.HoldUntil = t + 0.6f + (float)rng.NextDouble() * 0.6f;
                        }
                        if (s.Choice != 0 && ctx.Probe[s.Choice - 1] < 0.35f)
                            s.Choice = s.Choice == 3 ? 7 : 3;
                        cls = s.Choice;
                        jumpedNow = rng.NextDouble() < 0.03;
                    }
                    else if (best < 2500f)
                    {
                        cls = 2;
                    }
                    else
                    {
                        cls = 0;
                        crouch = true;
                    }
                    Vector2 move = cls == 0 ? Vector2.Zero : Features.FromFrame(Features.Direction(cls - 1), frame);
                    var flags = (SampleFlags)0;
                    if (s.Bot) flags |= SampleFlags.Bot;
                    if (s.Crouched) flags |= SampleFlags.Crouched;
                    if (s.Sprinting) flags |= SampleFlags.Sprinting;
                    if (s.Jumped) flags |= SampleFlags.Jumped;
                    if (enemy != null) flags |= SampleFlags.HasEnemy;
                    w.Write(new Sample
                    {
                        Time = t, Pid = s.Pid, Team = s.Team, Flags = flags, Health = s.Health,
                        Location = s.Loc, Velocity = s.Vel, Acceleration = new Vector3(move * Rate, 0),
                        AccelRate = Rate, GroundSpeed = Speed, Yaw = s.Yaw, Pitch = 0, FloorZ = s.Loc.Z - 42f,
                        Frame = Features.FrameOf(ctx), EnemyPid = enemy?.Pid ?? RecordingWriter.NoPid,
                        Weapon = weapon, Features = features,
                    });
                    ++total;

                    // Apply: stance and the jump show in the next sample; move and face.
                    s.Prev = move;
                    s.Crouched = crouch;
                    s.Sprinting = sprint;
                    s.Jumped = jumpedNow;
                    float speed = Speed * (crouch ? 0.4f : sprint ? 1.4f : 1f);
                    s.Vel = new Vector3(move * speed, 0);
                    s.Loc = Vector3.Clamp(s.Loc + s.Vel * Dt, new Vector3(30, 30, 0), new Vector3(Arena - 30, Arena - 30, 0));
                    if (enemy != null)
                        s.Yaw = MathF.Atan2(enemy.Loc.Y - s.Loc.Y, enemy.Loc.X - s.Loc.X);

                    if (enemy != null && best < 1500f && rng.NextDouble() < 0.002)
                    {
                        w.Kill(t, s.Pid, enemy.Pid, s.Loc + new Vector3(0, 0, 42), enemy.Loc + new Vector3(0, 0, 42), w.Intern("GCDmgType_Bullet"));
                        Respawn(enemy, rng);
                    }
                }
            }
        }
        Console.WriteLine($"wrote {files} recordings, {total} samples, to {dir}");
        return 0;
    }

    static void Respawn(Soldier s, Random rng)
    {
        s.Loc = new Vector3(200 + rng.NextSingle() * (Arena - 400), 200 + rng.NextSingle() * (Arena - 400), 42f);
        s.Vel = default;
        s.Yaw = rng.NextSingle() * MathF.Tau;
        s.Prev = default;
        s.EngagedSince = -1;
        s.Waypoint = new Vector2(s.Loc.X, s.Loc.Y);
    }
}

static class Check
{
    static int failures;

    static void Expect(bool ok, string what)
    {
        Console.WriteLine($"  {(ok ? "ok  " : "FAIL")} {what}");
        if (!ok) ++failures;
    }

    public static int Run(string policyPath, string parityPath)
    {
        var policy = Policy.Load(policyPath);
        Console.WriteLine($"policy: {policy.Describe}");
        Expect(policy.Schema == Features.Schema, "schema matches Features.Schema");

        using var doc = JsonDocument.Parse(File.ReadAllText(parityPath));
        var root = doc.RootElement;

        // 1. The forward pass agrees with numpy's.
        float worst = 0;
        int n = 0;
        var inputs = root.GetProperty("features").EnumerateArray();
        var expected = root.GetProperty("logits").EnumerateArray();
        var f = new float[policy.Inputs];
        while (inputs.MoveNext() && expected.MoveNext())
        {
            int i = 0;
            foreach (var v in inputs.Current.EnumerateArray()) f[i++] = v.GetSingle();
            var o = policy.Forward(f);
            int j = 0;
            foreach (var v in expected.Current.EnumerateArray())
                worst = MathF.Max(worst, MathF.Abs(o[j++] - v.GetSingle()));
            ++n;
        }
        Expect(n > 0 && worst < 1e-3f, $"forward pass on {n} vectors: worst |C# - numpy| = {worst:g3}");

        // 2. Movement classes: C# and the trainer label the same input the same way.
        int same = 0, cases = 0;
        foreach (var c in root.GetProperty("moves").EnumerateArray())
        {
            var v = new Vector2(c.GetProperty("fwd").GetSingle(), c.GetProperty("right").GetSingle());
            same += Features.MoveClass(v) == c.GetProperty("cls").GetInt32() ? 1 : 0;
            ++cases;
        }
        Expect(cases > 0 && same == cases, $"movement class convention: {same}/{cases} agree");

        // 3. It learned the script (only meaningful on synth recordings).
        if (root.TryGetProperty("synthetic", out var syn) && syn.GetBoolean())
        {
            Features.Context Ctx(float dist, Vector2 prev, bool crouched, bool sprinting, bool enemy = true)
            {
                var me = new Features.Soldier { Location = new Vector3(4500, 4500, 42), Yaw = 0, Health = 1 };
                return new Features.Context
                {
                    Self = me, HasEnemy = enemy,
                    Enemy = new Features.Soldier { Location = new Vector3(4500 + dist, 4500, 42), Yaw = MathF.PI, Health = 1 },
                    GroundSpeed = 440, VisibleEnemies = enemy ? 1 : 0, EngagedSeconds = enemy ? 2 : 0,
                    NearEnemies = enemy && dist < Features.NearEnemyRange ? 1 : 0,   // consistent with the distance
                    Probe = [1, 1, 1, 1, 1, 1, 1, 1], PreviousMove = prev, Crouched = crouched, Sprinting = sprinting,
                    WeaponGroup = 0,
                };
            }
            Policy.Decision D(Features.Context c)
            {
                var v = new float[Features.Count];
                Features.Write(c, v);
                return policy.Decide(v);
            }
            static int Arg(float[] p) { int b = 0; for (int i = 1; i < p.Length; ++i) if (p[i] > p[b]) b = i; return b; }

            var far = D(Ctx(3200, Vector2.Zero, true, false));
            Expect(Arg(far.Move) == 0 && far.Crouch > 0.8f, $"far enemy: hold still ({far.Move[0]:0.00}) and crouch ({far.Crouch:0.00})");
            var mid = D(Ctx(2000, Vector2.Normalize(new Vector2(1, 1)), false, false));   // already closing in
            Expect(Arg(mid.Move) == 2, $"mid range: forward-right diagonal ({mid.Move[2]:0.00})");
            var closing = D(Ctx(2000, Vector2.Zero, true, false));   // just came out of the crouch at long range
            Expect(closing.Crouch < 0.2f, $"mid range: stands up ({closing.Crouch:0.00})");
            var closeR = D(Ctx(900, new Vector2(0, 1), false, false));   // was strafing right (frame = +X, right = +Y)
            Expect(Arg(closeR.Move) == 3, $"close, was strafing right: keeps right ({closeR.Move[3]:0.00})");
            var closeL = D(Ctx(900, new Vector2(0, -1), false, false));
            Expect(Arg(closeL.Move) == 7, $"close, was strafing left: keeps left ({closeL.Move[7]:0.00})");
            var walled = Ctx(900, new Vector2(0, 1), false, false);
            // A flat wall on the right, 40 units away, as the arena's walls are: the 45 and 135 probes hit it too
            // (40 / cos 45 = 57 -> 0.14). The 90 probe alone short is a shape the recordings never have.
            walled.Probe = [1, 0.14f, 0.1f, 0.14f, 1, 1, 1, 1];
            var w = D(walled);
            Expect(w.Move[7] > w.Move[3], $"close, wall on the right: goes left ({w.Move[7]:0.00} > {w.Move[3]:0.00})");
            var none = D(Ctx(0, new Vector2(1, 0), false, true, enemy: false));
            Expect(none.Sprint > 0.8f && none.Crouch < 0.2f, $"no enemy: sprints ({none.Sprint:0.00}), stands ({none.Crouch:0.00})");
            Expect(closeR.Jump > far.Jump, $"jumps only up close ({closeR.Jump:0.000} > {far.Jump:0.000})");
        }

        Console.WriteLine(failures == 0 ? "all checks passed" : $"{failures} check(s) FAILED");
        return failures == 0 ? 0 : 1;
    }
}
