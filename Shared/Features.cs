// HomefrontImitation shared code. HomefrontBots carries a copy (HomefrontBots\Imitation\): keep the two identical.
// The schema string below goes into every recording and every trained policy; a bot refuses a policy whose schema
// isn't its own, so a change here means new recordings.
using System;
using System.Numerics;

namespace HomefrontImitation;

/// <summary>The observation a soldier acts on, as one fixed-length vector: the same function describes a recorded
/// human (HomefrontImitation) and a bot asking the policy what to do (HomefrontBots). Everything is egocentric, in
/// the "frame": a flat unit vector toward the enemy being fought, or along the view when there is none. Directions
/// in the frame are angles from its forward axis toward its right, in 45 degree steps (<see cref="Direction"/>).
/// No SDK types: this file builds and is tested outside the game.</summary>
public static class Features
{
    public const int Version = 1;

    /// <summary>Samples per second, in recordings and for bot decisions.</summary>
    public const float SampleSeconds = 0.1f;

    public const float SightRange = 4000f;   // as far as GCMPBot engages soldiers (MaxTargetDistSoldier)
    public const float AllyRange = 1500f;
    public const float NearEnemyRange = 2500f;
    public const float ProbeLength = 400f;   // free-space probes, from knee height
    public const int Probes = 8;
    public const int WeaponGroups = 5;

    /// <summary>Below this fraction of AccelRate the soldier counts as standing still.</summary>
    public const float StillThreshold = 0.3f;

    /// <summary>Movement classes: 0 = still, 1..8 = directions 0, 45, ..., 315 degrees in the frame.</summary>
    public const int MoveClasses = 9;

    public static readonly string[] Names =
    [
        "has_enemy", "enemy_dist", "enemy_dz",
        "vel_fwd", "vel_right", "enemy_vel_fwd", "enemy_vel_right",
        "enemy_facing_me", "aim_on_frame",
        "health", "enemy_health",
        "visible_enemies", "near_allies", "near_enemies", "engaged",
        "probe_0", "probe_45", "probe_90", "probe_135", "probe_180", "probe_225", "probe_270", "probe_315",
        "prev_fwd", "prev_right",
        "crouched", "sprinting", "falling",
        "wpn_rifle", "wpn_smg", "wpn_lmg", "wpn_sniper", "wpn_other",
    ];

    public static int Count => Names.Length;

    /// <summary>Identifies the feature layout: recordings and policies carry it, bots check it.</summary>
    public static string Schema => $"v{Version}:" + string.Join(",", Names);

    /// <summary>One soldier as the features see it. Yaw in radians (Unreal yaw * PI / 32768).</summary>
    public struct Soldier
    {
        public Vector3 Location, Velocity;
        public float Yaw;
        public float Health;  // fraction of HealthMax
    }

    /// <summary>Everything <see cref="Write"/> needs.</summary>
    public struct Context
    {
        public Soldier Self;
        public bool HasEnemy;
        public Soldier Enemy;
        public float GroundSpeed;
        public int VisibleEnemies, NearAllies, NearEnemies;
        public float EngagedSeconds;         // how long some enemy has been in sight without a break
        public float[] Probe;                // Probes fractions (0 blocked at once .. 1 free), in frame order
        public Vector2 PreviousMove;         // the last sample's movement input (world, flat, |v| <= 1)
        public bool Crouched, Sprinting, Falling;
        public int WeaponGroup;              // WeaponGroupOf
    }

    /// <summary>The frame: toward the enemy (flat) when there is one, else along the view.</summary>
    public static Vector2 FrameOf(in Context c)
    {
        if (c.HasEnemy)
        {
            var d = new Vector2(c.Enemy.Location.X - c.Self.Location.X, c.Enemy.Location.Y - c.Self.Location.Y);
            if (d.LengthSquared() > 1f)
                return Vector2.Normalize(d);
        }
        return YawDir(c.Self.Yaw);
    }

    public static Vector2 YawDir(float yaw) => new(MathF.Cos(yaw), MathF.Sin(yaw));

    /// <summary>The frame's right axis (Unreal: X forward, Y right, Z up).</summary>
    public static Vector2 Right(Vector2 frame) => new(-frame.Y, frame.X);

    /// <summary>A flat world vector in the frame: (forward, right).</summary>
    public static Vector2 ToFrame(Vector2 v, Vector2 frame) => new(Vector2.Dot(v, frame), Vector2.Dot(v, Right(frame)));

    /// <summary>A frame vector (forward, right) in the world.</summary>
    public static Vector2 FromFrame(Vector2 f, Vector2 frame) => frame * f.X + Right(frame) * f.Y;

    /// <summary>Unit direction of angle index <paramref name="i"/> (0..7, 45 degree steps) in the frame.</summary>
    public static Vector2 Direction(int i)
    {
        float a = i * MathF.PI / 4f;
        return new Vector2(MathF.Cos(a), MathF.Sin(a));
    }

    /// <summary>The movement class of an input (frame coordinates, |v| = fraction of AccelRate).</summary>
    public static int MoveClass(Vector2 inFrame)
    {
        if (inFrame.Length() < StillThreshold)
            return 0;
        float a = MathF.Atan2(inFrame.Y, inFrame.X);
        int i = (int)MathF.Round(a / (MathF.PI / 4f));
        return ((i % 8) + 8) % 8 + 1;
    }

    /// <summary>The weapon group of a weapon (its WeaponInfo or class name): 0 rifle, 1 SMG, 2 LMG, 3 sniper, 4 other.</summary>
    public static int WeaponGroupOf(string? name)
    {
        if (string.IsNullOrEmpty(name))
            return 4;
        string n = name.ToUpperInvariant();
        if (n.Contains("SNI") || n.Contains("SNP") || n.Contains("M110") || n.Contains("M200") || n.Contains("SVD"))
            return 3;
        if (n.Contains("LMG") || n.Contains("M249") || n.Contains("SCAR-MG") || n.Contains("_MG") || n.Contains("PKP"))
            return 2;
        if (n.Contains("SMG") || n.Contains("DIABLO") || n.Contains("PWS") || n.Contains("VEPR") || n.Contains("MP7"))
            return 1;
        if (n.Contains("RIF") || n.Contains("M4") || n.Contains("ACR") || n.Contains("SCAR") || n.Contains("M16") || n.Contains("AK") || n.Contains("XM10") || n.Contains("T3AK"))
            return 0;
        return 4;
    }

    static float Clamp(float v, float lo, float hi) => v < lo ? lo : v > hi ? hi : v;

    /// <summary>Writes the vector for <paramref name="c"/> into <paramref name="f"/> (length <see cref="Count"/>).</summary>
    public static void Write(in Context c, Span<float> f)
    {
        if (f.Length < Count)
            throw new ArgumentException($"need {Count} floats", nameof(f));
        f[..Count].Clear();
        Vector2 frame = FrameOf(c);
        float speed = c.GroundSpeed > 1f ? c.GroundSpeed : 440f;
        var selfVel = ToFrame(new Vector2(c.Self.Velocity.X, c.Self.Velocity.Y), frame) / speed;
        int k = 0;
        f[k++] = c.HasEnemy ? 1f : 0f;
        if (c.HasEnemy)
        {
            var d = c.Enemy.Location - c.Self.Location;
            float flat = new Vector2(d.X, d.Y).Length();
            var enemyVel = ToFrame(new Vector2(c.Enemy.Velocity.X, c.Enemy.Velocity.Y), frame) / speed;
            // The enemy's view against the line from it to me: 1 = looking straight at me.
            float facing = Vector2.Dot(YawDir(c.Enemy.Yaw), -frame);
            f[k++] = Clamp(flat / SightRange, 0f, 1.5f);
            f[k++] = Clamp(d.Z / 500f, -2f, 2f);
            f[k++] = Clamp(selfVel.X, -2f, 2f);
            f[k++] = Clamp(selfVel.Y, -2f, 2f);
            f[k++] = Clamp(enemyVel.X, -2f, 2f);
            f[k++] = Clamp(enemyVel.Y, -2f, 2f);
            f[k++] = facing;
        }
        else
        {
            k += 2;
            f[k++] = Clamp(selfVel.X, -2f, 2f);
            f[k++] = Clamp(selfVel.Y, -2f, 2f);
            k += 3;
        }
        f[k++] = Vector2.Dot(YawDir(c.Self.Yaw), frame);
        f[k++] = Clamp(c.Self.Health, 0f, 1f);
        f[k++] = c.HasEnemy ? Clamp(c.Enemy.Health, 0f, 1f) : 0f;
        f[k++] = Math.Min(c.VisibleEnemies, 4) / 4f;
        f[k++] = Math.Min(c.NearAllies, 4) / 4f;
        f[k++] = Math.Min(c.NearEnemies, 4) / 4f;
        f[k++] = Clamp(c.EngagedSeconds / 5f, 0f, 1f);
        for (int i = 0; i < Probes; ++i)
            f[k++] = c.Probe != null && i < c.Probe.Length ? Clamp(c.Probe[i], 0f, 1f) : 1f;
        var prev = ToFrame(c.PreviousMove, frame);
        f[k++] = Clamp(prev.X, -1f, 1f);
        f[k++] = Clamp(prev.Y, -1f, 1f);
        f[k++] = c.Crouched ? 1f : 0f;
        f[k++] = c.Sprinting ? 1f : 0f;
        f[k++] = c.Falling ? 1f : 0f;
        f[k + Math.Clamp(c.WeaponGroup, 0, WeaponGroups - 1)] = 1f;
        k += WeaponGroups;
        if (k != Count)
            throw new InvalidOperationException($"feature layout wrote {k} of {Count}");
    }
}
