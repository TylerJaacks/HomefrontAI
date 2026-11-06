// HomefrontImitation shared code. HomefrontBots carries a copy (HomefrontBots\Imitation\): keep the two identical.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace HomefrontImitation;

/// <summary>The trained policy (policy.json, written by train\hfimitate.py): a small MLP from the
/// <see cref="Features"/> vector to what a human did next. Heads: movement class (softmax over
/// <see cref="Features.MoveClasses"/>), and the chances of being crouched, sprinting, and jumping in the next
/// sample. Plain arrays, no dependencies: a forward pass is a few thousand multiply-adds.</summary>
public sealed class Policy
{
    public const string Format = "homefront-imitation-policy";

    sealed class Layer
    {
        public int In, Out;
        public float[] W = [];   // Out x In, row-major
        public float[] B = [];
        public bool Relu;
    }

    readonly List<Layer> layers = new();
    float[] mean = [], std = [];
    int moveOffset, crouchAt, sprintAt, jumpAt;

    public string Schema { get; private set; } = "";
    public string Trained { get; private set; } = "";
    public long Samples { get; private set; }
    public int Inputs => mean.Length;
    public int Outputs { get; private set; }
    public string Describe => $"{Samples} samples, trained {Trained}, {layers.Count} layers";

    /// <summary>Loads policy.json. Throws on a malformed file; check <see cref="Schema"/> against
    /// <see cref="Features.Schema"/> before using it.</summary>
    public static Policy Load(string path) => Parse(File.ReadAllText(path));

    public static Policy Parse(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (root.GetProperty("format").GetString() != Format)
            throw new InvalidDataException("not a HomefrontImitation policy");
        var p = new Policy
        {
            Schema = root.GetProperty("schema").GetString() ?? "",
            Trained = root.TryGetProperty("trained_utc", out var t) ? t.GetString() ?? "" : "",
            Samples = root.TryGetProperty("samples", out var s) ? s.GetInt64() : 0,
            mean = Floats(root.GetProperty("mean")),
            std = Floats(root.GetProperty("std")),
        };
        int width = p.mean.Length;
        if (p.std.Length != width)
            throw new InvalidDataException("mean/std length mismatch");
        foreach (var l in root.GetProperty("layers").EnumerateArray())
        {
            var layer = new Layer
            {
                In = l.GetProperty("in").GetInt32(),
                Out = l.GetProperty("out").GetInt32(),
                W = Floats(l.GetProperty("w")),
                B = Floats(l.GetProperty("b")),
                Relu = l.GetProperty("act").GetString() == "relu",
            };
            if (layer.In != width || layer.W.Length != layer.In * layer.Out || layer.B.Length != layer.Out)
                throw new InvalidDataException($"layer {p.layers.Count}: shape mismatch");
            width = layer.Out;
            p.layers.Add(layer);
        }
        if (p.layers.Count == 0)
            throw new InvalidDataException("no layers");
        p.Outputs = width;
        var heads = root.GetProperty("heads");
        p.moveOffset = heads.GetProperty("move").GetProperty("offset").GetInt32();
        int moveCount = heads.GetProperty("move").GetProperty("count").GetInt32();
        p.crouchAt = heads.GetProperty("crouch").GetInt32();
        p.sprintAt = heads.GetProperty("sprint").GetInt32();
        p.jumpAt = heads.GetProperty("jump").GetInt32();
        if (moveCount != Features.MoveClasses || p.moveOffset + moveCount > width ||
            Math.Max(p.crouchAt, Math.Max(p.sprintAt, p.jumpAt)) >= width)
            throw new InvalidDataException("heads don't fit the output layer");
        return p;
    }

    static float[] Floats(JsonElement e)
    {
        var a = new float[e.GetArrayLength()];
        int i = 0;
        foreach (var v in e.EnumerateArray())
            a[i++] = v.GetSingle();
        return a;
    }

    /// <summary>What the policy says for <paramref name="features"/>.</summary>
    public struct Decision
    {
        public float[] Move;   // MoveClasses probabilities (temperature applied)
        public float Crouch, Sprint, Jump;
    }

    float[] bufA = [], bufB = [];

    /// <summary>The raw outputs (logits) for one feature vector. Not thread-safe (the game thread only).</summary>
    public ReadOnlySpan<float> Forward(ReadOnlySpan<float> features)
    {
        if (features.Length < Inputs)
            throw new ArgumentException($"need {Inputs} features");
        int max = Inputs;
        foreach (var l in layers)
            max = Math.Max(max, l.Out);
        if (bufA.Length < max)
        {
            bufA = new float[max];
            bufB = new float[max];
        }
        float[] x = bufA, y = bufB;
        for (int i = 0; i < Inputs; ++i)
            x[i] = (features[i] - mean[i]) / std[i];
        foreach (var l in layers)
        {
            for (int o = 0; o < l.Out; ++o)
            {
                float sum = l.B[o];
                int row = o * l.In;
                for (int i = 0; i < l.In; ++i)
                    sum += l.W[row + i] * x[i];
                y[o] = l.Relu && sum < 0f ? 0f : sum;
            }
            (x, y) = (y, x);
        }
        return x.AsSpan(0, Outputs);
    }

    /// <summary>The decision for one feature vector. <paramref name="temperature"/> &lt; 1 sharpens the movement
    /// choice toward the likeliest, &gt; 1 flattens it.</summary>
    public Decision Decide(ReadOnlySpan<float> features, float temperature = 1f)
    {
        var o = Forward(features);
        var d = new Decision { Move = new float[Features.MoveClasses] };
        float t = temperature > 0.05f ? temperature : 0.05f, top = float.MinValue, sum = 0f;
        for (int i = 0; i < Features.MoveClasses; ++i)
            top = Math.Max(top, o[moveOffset + i] / t);
        for (int i = 0; i < Features.MoveClasses; ++i)
            sum += d.Move[i] = MathF.Exp(o[moveOffset + i] / t - top);
        for (int i = 0; i < Features.MoveClasses; ++i)
            d.Move[i] /= sum;
        d.Crouch = Sigmoid(o[crouchAt]);
        d.Sprint = Sigmoid(o[sprintAt]);
        d.Jump = Sigmoid(o[jumpAt]);
        return d;
    }

    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>A movement class drawn from <paramref name="move"/>, skipping classes with <paramref name="allowed"/>
    /// false (blocked directions). -1 when nothing allowed has any weight.</summary>
    public static int Sample(float[] move, Func<int, bool> allowed, Random random)
    {
        float total = 0f;
        for (int i = 0; i < move.Length; ++i)
            if (allowed(i))
                total += move[i];
        if (total <= 1e-6f)
            return -1;
        float r = random.NextSingle() * total;
        int last = -1;
        for (int i = 0; i < move.Length; ++i)
        {
            if (!allowed(i))
                continue;
            last = i;
            r -= move[i];
            if (r <= 0f)
                return i;
        }
        return last;
    }
}
