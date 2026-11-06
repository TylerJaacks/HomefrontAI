// HomefrontAI shared code. HomefrontBots carries a copy (HomefrontBots\Imitation\): keep the two identical.
// The game-side half of Features: reads a soldier, its enemies and its surroundings out of the running game.
using System;
using System.Collections.Generic;
using System.Numerics;
using HomefrontSDK;

namespace HomefrontAI;

/// <summary>Builds <see cref="Features.Context"/> for soldiers in the game, the same way for a recorded human and for
/// a bot: who is in sight (eye to eye, world geometry only), how many allies and enemies are near, how much room
/// there is around it (eight short box sweeps at knee height, in the frame), and what it did last sample. Keeps a
/// little memory per soldier (since when it has been engaged, its last movement input).</summary>
public sealed class Senses
{
    sealed class Memory
    {
        public float EngagedSince = -1f;
        public Vector2 Move;
        public float Seen;
    }

    readonly Dictionary<nint, Memory> memory = new();
    readonly List<(float Dist, Pawn Pawn)> near = new();
    readonly float[] probe = new float[Features.Probes];
    float lastPrune;

    public long Traces { get; private set; }

    public static Vector3 V(FVector v) => new(v.X, v.Y, v.Z);
    public static FVector F(Vector3 v) => new(v.X, v.Y, v.Z);

    public static float YawOf(Pawn pawn) =>
        (pawn.Controller is Controller c && c.IsValid ? c.Rotation.Yaw : pawn.Rotation.Yaw) * MathF.PI / 32768f;

    public static float PitchOf(Pawn pawn)
    {
        int p = (pawn.Controller is Controller c && c.IsValid ? c.Rotation.Pitch : pawn.Rotation.Pitch) & 0xFFFF;
        return (p > 32767 ? p - 65536 : p) * MathF.PI / 32768f;
    }

    public static float HealthOf(Pawn p) => p.HealthMax > 0 ? Math.Clamp((float)p.Health / p.HealthMax, 0f, 1f) : 0f;

    public static float HalfHeight(Pawn p) => p.CylinderComponent is CylinderComponent cyl && cyl.CollisionHeight > 1f ? cyl.CollisionHeight : 42f;

    public static FVector Eye(Pawn p)
    {
        FVector l = p.Location;
        return new FVector(l.X, l.Y, l.Z + p.BaseEyeHeight);
    }

    /// <summary>Its weapon's config name (CFG_RIF_M4-like) or class, for <see cref="Features.WeaponGroupOf"/>.</summary>
    public static string WeaponName(Pawn p)
    {
        Weapon? w = p.Weapon;
        if (w == null || !w.IsValid)
            return "";
        if (w is GCWeaponBase gw && gw.WeaponInfo is GCWeaponInfo info && info.IsValid)
            return info.ObjectName;
        return w.ClassName;
    }

    public static Features.Soldier SoldierOf(Pawn p) => new()
    {
        Location = V(p.Location),
        Velocity = V(p.Velocity),
        Yaw = YawOf(p),
        Health = HealthOf(p),
    };

    static bool Clear(World world, Actor source, FVector from, FVector to)
    {
        try
        {
            return !world.Trace(source, from, to, default).Hit;
        }
        catch (GameCrashException)
        {
            return false;
        }
    }

    /// <summary>The movement input a soldier used this sample (world, flat, |v| &lt;= 1), for its next
    /// "prev_fwd/prev_right".</summary>
    public void SetMove(Pawn self, Vector2 move)
    {
        if (self.Controller is Controller c)
            Get(c.Ptr, 0f).Move = move.LengthSquared() > 1f ? Vector2.Normalize(move) : move;
    }

    Memory Get(nint key, float now)
    {
        if (!memory.TryGetValue(key, out var m))
            memory[key] = m = new Memory();
        if (now > 0f)
            m.Seen = now;
        return m;
    }

    /// <summary>Forgets everyone (map change).</summary>
    public void Clear()
    {
        memory.Clear();
        lastPrune = 0f;
    }

    /// <summary>Observes <paramref name="self"/>. <paramref name="soldiers"/> is every living soldier with its team;
    /// <paramref name="preferred"/> (a bot's AI enemy) is chosen over the nearest visible enemy when it's in sight.
    /// <paramref name="now"/> is in seconds. Returns the enemy in the frame, or null.</summary>
    public Pawn? Observe(Pawn self, int team, IReadOnlyList<(Pawn Pawn, int Team)> soldiers, Pawn? preferred,
        World world, Actor traceSource, float now, out Features.Context ctx)
    {
        if (now - lastPrune > 30f)
        {
            lastPrune = now;
            var stale = new List<nint>();
            foreach (var (k, m) in memory)
                if (now - m.Seen > 30f)
                    stale.Add(k);
            foreach (nint k in stale)
                memory.Remove(k);
        }

        Memory mem = Get(self.Controller?.Ptr ?? self.Ptr, now);
        FVector eye = Eye(self), at = self.Location;
        near.Clear();
        int allies = 0, enemies = 0;
        foreach (var (other, otherTeam) in soldiers)
        {
            if (other.Ptr == self.Ptr)
                continue;
            float d = FVector.Dist(at, other.Location);
            if (otherTeam == team)
            {
                if (d < Features.AllyRange)
                    ++allies;
                continue;
            }
            if (d < Features.NearEnemyRange)
                ++enemies;
            if (d < Features.SightRange)
                near.Add((other.Ptr == preferred?.Ptr ? -1f : d, other));  // the preferred one is checked first
        }
        near.Sort((a, b) => a.Dist.CompareTo(b.Dist));

        Pawn? enemy = null;
        int visible = 0;
        for (int i = 0; i < near.Count && i < 4; ++i)
        {
            ++Traces;
            if (!Clear(world, traceSource, eye, Eye(near[i].Pawn)))
                continue;
            ++visible;
            enemy ??= near[i].Pawn;
        }

        if (enemy == null)
            mem.EngagedSince = -1f;
        else if (mem.EngagedSince < 0f)
            mem.EngagedSince = now;

        ctx = new Features.Context
        {
            Self = SoldierOf(self),
            HasEnemy = enemy != null,
            GroundSpeed = self.GroundSpeed,
            VisibleEnemies = visible,
            NearAllies = allies,
            NearEnemies = enemies,
            EngagedSeconds = mem.EngagedSince < 0f ? 0f : now - mem.EngagedSince,
            PreviousMove = mem.Move,
            Crouched = self.bIsCrouched,
            Sprinting = self.bIsSprinting,
            Falling = self.Physics == EPhysics.PHYS_Falling,
            WeaponGroup = Features.WeaponGroupOf(WeaponName(self)),
            Probe = probe,
        };
        if (enemy != null)
            ctx.Enemy = SoldierOf(enemy);

        // Room to move: box sweeps at knee height (above a step), each direction of the frame.
        Vector2 frame = Features.FrameOf(ctx);
        float knee = at.Z - HalfHeight(self) + 50f;
        var start = new FVector(at.X, at.Y, knee);
        var extent = new FVector(10f, 10f, 10f);
        for (int i = 0; i < Features.Probes; ++i)
        {
            Vector2 dir = Features.FromFrame(Features.Direction(i), frame);
            var end = new FVector(at.X + dir.X * Features.ProbeLength, at.Y + dir.Y * Features.ProbeLength, knee);
            ++Traces;
            try
            {
                TraceHit hit = world.Trace(traceSource, start, end, extent);
                probe[i] = hit.Hit ? Math.Clamp(hit.Time, 0f, 1f) : 1f;
            }
            catch (GameCrashException)
            {
                probe[i] = 0f;
            }
        }
        ctx.Probe = (float[])probe.Clone();
        return enemy;
    }
}
