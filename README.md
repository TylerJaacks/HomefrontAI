# HomefrontImitation - learn bot behaviour from real players

A [HomefrontSDK](https://github.com/TylerJaacks/HomefrontSDK) mod for the Homefront dedicated server that records
how real players move and fight, and a trainer that turns the recordings into a model
[HomefrontBots](https://github.com/TylerJaacks/HomefrontBots) imports: bots then move in fights, crouch, sprint and
jump the way humans do in the same situation, prefer the routes humans walk, and hold the spots humans win from.

```
 server (HomefrontImitation)            your PC or the server (train\hfimitate.py)          server (HomefrontBots)
 10 Hz samples of every soldier  ──►    behaviour cloning (MLP, numpy)        ──►   policy.json   fight movement, stance
 + kills, fire, jumps (.hfrec)          per-map floor heat (48-unit cells)    ──►   <map>.heat    routes, objective spots
```

## What it learns

| Part | From | Used by HomefrontBots for |
|---|---|---|
| **policy** | A 33-value observation per soldier (below) -> what the player did next: which way it moved in the enemy's frame (still or one of 8 directions), and whether it was crouched, sprinting or jumped in the next sample | **combat**: in `Attacking`, each 0.1 s, the bot samples a move from the policy (directions with a wall in the way left out) and walks to a nav node that way along a tested straight line. **stance**: crouch/sprint/jump |
| **heat** | Per map, per floor cell of the nav grid: seconds humans spent there, seconds there in a fight, kills made from there, deaths there | **routes**: A* adds a cost (default 0.6 x distance) on floor humans never use. **spots**: at objectives (and roaming), spots drawn by where humans hold and win |

The observation (`Shared/Features.cs`, the same code in the recorder and in the bots) is egocentric, in a "frame"
pointing at the enemy in sight (or along the view without one): enemy distance and height, both soldiers' velocities
in the frame, whether the enemy faces me, health, how many enemies are visible/near and allies near, how long the
fight has lasted, eight free-space probes around (box sweeps at knee height), last sample's movement, stance, and the
weapon class (rifle, SMG, LMG, sniper, other). Aiming and firing stay with the game's own `GCWeaponAIController`.

## Install

Stop the server. From this repository's GitHub Releases put `HomefrontImitation.dll` in
`Homefront Dedicated Server\Binaries\HomefrontSDK\Mods\HomefrontImitation\`, and a HomefrontBots build with imitation
support (`Imitation\` in its source) in its own folder. Or build from source, the repositories side by side as HomefrontBots' README
describes, plus `HomefrontImitation\` beside them:

```
dotnet build HomefrontImitation\HomefrontImitation.csproj -c Release
HomefrontSDK\Build.ps1 -Install          installs it with the SDK and the other mods
```

Needs the 32-bit (x86) .NET 10 runtime on the server, like every SDK mod. The trainer needs Python 3.10+ and numpy
(`pip install -r HomefrontImitation\train\requirements.txt`).

## Use

1. **Record.** Play on the server with people. Every map gets its own file in
   `Mods\HomefrontImitation\recordings\<map>_<utc time>.hfrec`, about 200 bytes per soldier per sample (32 players:
   ~200 MB an hour). Bots are recorded too, flagged, so `stats` can compare them with the humans; training leaves
   them out.
2. **Train** (on the server machine, or copy the recordings anywhere):
   ```
   python HomefrontImitation\train\hfimitate.py stats      what's recorded; humans vs bots: K/D, time in fights, strafing, stance
   python HomefrontImitation\train\hfimitate.py train      -> Mods\HomefrontBots\imitation\policy.json and <map>.heat
   ```
   Both default to the folders above; give recordings and `--out` to point elsewhere, `--maps fl-farm FL-Spillway`
   to pick maps. It prints how much better than guessing the policy predicts the held-out recordings.
3. **Load** with `!bots imitation reload` (or the next map). `!bots imitation` shows what's in use.
4. **Compare.** Keep recording: `stats` now shows how the bots behave against the humans. Switch parts off
   (`!bots imitation combat off`) to compare against the hand-written behaviour.

A map's heat is only used once humans spent 10 minutes there (`minHeatMinutes`); until then bots use the old
spot choice and plain shortest paths. The policy is shared across maps.

## Commands

`!rec` (the recorder; changing it needs an admin or the bridge):

| Command | What |
|---|---|
| `!rec` / `status` | the file being written, samples, kills, size |
| `on` / `off` | start/stop recording (saved) |
| `bots on` / `bots off` | record bots too (default on) |
| `split` | close this file and start a new one |

`!bots imitation` (in HomefrontBots):

| Command | What |
|---|---|
| `!bots imitation` | which parts are in use, the policy, this map's heat |
| `on` / `off` | everything learned on/off |
| `combat`, `stance`, `routes`, `spots` `on`/`off` | one part |
| `temp <0.1-2>` | policy temperature: below 1 closer to the likeliest human move, above 1 more varied (default 0.8) |
| `reload` | read policy.json, this map's heat and imitation.json again |

Bridge requests: `imitation.status` (recorder), `bots.imitation` (bots).

## Settings

`Mods\HomefrontImitation\imitation.json` (written with the defaults on first start):

| Key | Default | |
|---|---|---|
| `enabled` | true | record |
| `recordBots` | true | bots too |
| `minHumans` | 1 | record only with at least this many humans in play |
| `announce` | true | tell joining players movement is recorded (anonymously) |
| `maxRecordingMB` | 4096 | stop when the recordings folder holds this much |
| `maxSoldiers` | 64 | per sample |

`Mods\HomefrontBots\imitation\imitation.json`: `enabled`, `combat`, `stance`, `routes`, `spots`, `temperature`
(0.8), `routeBias` (0.6), `fightSpeed` (0.85 of GroundSpeed), `minHeatMinutes` (10).

## Privacy

Recordings hold no player names, Steam IDs or chat: each soldier is a number per recording plus its team and
whether it's a bot. Positions, movement input, view direction, stance, health, weapon class, fire/jump/kill events.
Joining players are told (turn `announce` off if you tell them another way).

## Source

| Path | What |
|---|---|
| `Recorder.cs` | The script: sampling at 10 Hz, hooks (`Weapon.ServerStartFire/StopFire`, `Pawn.DoJump`, `GameInfo.Killed`), files, `!rec` |
| `Recording/RecordingWriter.cs` | The `.hfrec` format (documented in the file and in `train/hfrec.py`) |
| `Shared/Features.cs` | The observation and the movement-class convention. **Copied into HomefrontBots\Imitation**: keep identical |
| `Shared/Senses.cs` | Reading it out of the game (sight lines, probes, memory per soldier). Copied too |
| `Shared/Policy.cs` | Loads policy.json and runs it (no dependencies). Copied too |
| `train/hfrec.py`, `train/hfimitate.py` | Reader and trainer (numpy only) |
| `tests/` | `python tests/e2e.py`: the harness writes scripted recordings with the real writer and features, the trainer learns them, and C# checks its forward pass against numpy's (to 1e-5), the label convention, that the script was learned, the heat format, and that HomefrontBots' copies match |

Recordings and policies carry the feature schema (`Features.Schema`). HomefrontBots refuses a policy trained on
another schema, and the trainer uses only recordings with the newest one: change `Features.cs`, bump
`Features.Version`, record again.

## Limits

- **Not tried on a live server yet.** The pipeline is tested end to end outside the game; in game, check first that
  `Pawn.Acceleration` of human players holds their input (it's what `ServerMove` sets), and that `ShouldCrouch` /
  `SetSprinting` work on bot pawns.
- Movement is learned in fights only. Out of fights humans' goals are unknown, so where they go is learned as heat,
  not as a policy.
- Learns movement, not aim: the game's AI still aims and fires (its reaction and accuracy are untouched).
- One policy for all maps and game modes; the heat is per map.
- Behaviour cloning copies the players you record, good and bad. More players, more maps, better model.
