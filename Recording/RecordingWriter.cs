using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace HomefrontAI;

/// <summary>Sample flags (<see cref="Sample.Flags"/>).</summary>
[Flags]
public enum SampleFlags : ushort
{
    Bot = 1 << 0,
    Crouched = 1 << 1,
    Sprinting = 1 << 2,
    Falling = 1 << 3,
    Ladder = 1 << 4,
    Firing = 1 << 5,
    Jumped = 1 << 6,     // a jump started since the soldier's previous sample
    HasEnemy = 1 << 7,
    Walking = 1 << 8,
}

/// <summary>One soldier at one sample time.</summary>
public struct Sample
{
    public float Time;            // seconds since the recording started
    public ushort Pid;            // this recording's id for the soldier (not its name)
    public sbyte Team;
    public SampleFlags Flags;
    public float Health;          // fraction of HealthMax
    public Vector3 Location, Velocity, Acceleration;
    public float AccelRate, GroundSpeed;
    public float Yaw, Pitch;      // radians
    public float FloorZ;          // Location.Z minus the collision half height
    public Vector2 Frame;         // the features' frame (Features.FrameOf)
    public ushort EnemyPid;       // NoPid without an enemy in sight
    public ushort Weapon;         // string table id
    public float[] Features;      // Features.Count
}

/// <summary>The recording format (.hfrec, little-endian), read by train\hfrec.py. Header: "HFRC", format version
/// (u16), feature version (u16), feature count (u16), sample seconds (f32), start (unix seconds, i64), then strings
/// (u16 length + UTF-8) map, game mode, feature schema. Then records, each a tag byte:
/// 1 sample, 2 kill, 3 string table entry, 4 soldier (first seen), 5 end of recording.</summary>
public sealed class RecordingWriter : IDisposable
{
    public const uint Magic = 0x43524648;   // "HFRC"
    public const ushort FormatVersion = 1;
    public const ushort NoPid = 0xFFFF;

    const byte TagSample = 1, TagKill = 2, TagString = 3, TagSoldier = 4, TagEnd = 5;

    readonly BinaryWriter w;
    readonly Dictionary<string, ushort> strings = new(StringComparer.Ordinal);

    public string Path { get; }
    public long Bytes => w.BaseStream.Position;
    public long SampleCount { get; private set; }
    public long KillCount { get; private set; }

    public RecordingWriter(string path, string map, string mode, DateTimeOffset start)
    {
        Path = path;
        var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read, 1 << 16);
        w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: false);
        w.Write(Magic);
        w.Write(FormatVersion);
        w.Write((ushort)Features.Version);
        w.Write((ushort)Features.Count);
        w.Write(Features.SampleSeconds);
        w.Write(start.ToUnixTimeSeconds());
        Str(map);
        Str(mode);
        Str(Features.Schema);
    }

    void Str(string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        if (b.Length > ushort.MaxValue)
            Array.Resize(ref b, ushort.MaxValue);
        w.Write((ushort)b.Length);
        w.Write(b);
    }

    /// <summary>The id of a string (weapon, damage type), written to the table the first time.</summary>
    public ushort Intern(string? s)
    {
        s ??= "";
        if (strings.TryGetValue(s, out ushort id))
            return id;
        id = (ushort)Math.Min(strings.Count, ushort.MaxValue - 1);
        strings[s] = id;
        w.Write(TagString);
        w.Write(id);
        Str(s);
        return id;
    }

    /// <summary>A soldier seen for the first time: its id, team and whether it's a bot. No name or Steam id.</summary>
    public void Soldier(ushort pid, sbyte team, bool bot)
    {
        w.Write(TagSoldier);
        w.Write(pid);
        w.Write(team);
        w.Write(bot ? (byte)1 : (byte)0);
    }

    static void V3(BinaryWriter w, Vector3 v)
    {
        w.Write(v.X);
        w.Write(v.Y);
        w.Write(v.Z);
    }

    public void Write(in Sample s)
    {
        if (s.Features == null || s.Features.Length != Features.Count)
            throw new ArgumentException($"a sample needs {Features.Count} features");
        w.Write(TagSample);
        w.Write(s.Time);
        w.Write(s.Pid);
        w.Write(s.Team);
        w.Write((ushort)s.Flags);
        w.Write(s.Health);
        V3(w, s.Location);
        V3(w, s.Velocity);
        V3(w, s.Acceleration);
        w.Write(s.AccelRate);
        w.Write(s.GroundSpeed);
        w.Write(s.Yaw);
        w.Write(s.Pitch);
        w.Write(s.FloorZ);
        w.Write(s.Frame.X);
        w.Write(s.Frame.Y);
        w.Write(s.EnemyPid);
        w.Write(s.Weapon);
        foreach (float f in s.Features)
            w.Write(f);
        ++SampleCount;
    }

    public void Kill(float time, ushort killer, ushort victim, Vector3 killerAt, Vector3 victimAt, ushort damageType)
    {
        w.Write(TagKill);
        w.Write(time);
        w.Write(killer);
        w.Write(victim);
        V3(w, killerAt);
        V3(w, victimAt);
        w.Write(damageType);
        ++KillCount;
    }

    public void Flush() => w.Flush();

    public void Dispose()
    {
        try
        {
            w.Write(TagEnd);
            w.Flush();
        }
        catch (IOException)
        {
        }
        try
        {
            w.Dispose();
        }
        catch (IOException)
        {
        }
    }
}
