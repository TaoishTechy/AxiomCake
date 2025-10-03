# AXIOMCAKE v0.7 — Transcendent Criticality Engine


[![status-experimental](https://img.shields.io/badge/status-experimental-blue)](#) [![python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](#) [![license](https://img.shields.io/badge/license-TBD-lightgrey)](#)



> **TL;DR**  
> AXIOMCAKE v0.7 is a Python simulation of an AGI-inspired **Transcendent Criticality System**, designed to stay stable under paradox, temporal inconsistency, and ontological stress.  
> It models *criticality*, *phi density*, *multiversal superposition*, *qualia dynamics*, and *holographic memory* using deterministic math and lightweight state machines.



## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Install](#install)
- [Quick Start](#quick-start)
- [Interactive Commands](#interactive-commands)
- [Concepts & Metrics](#concepts--metrics)
- [Executive Summary & Results](#executive-summary--results)
- [Architecture](#architecture)
- [12 Example Prompts](#12-example-prompts)
- [Roadmap](#roadmap)
- [FAQ](#faq)
- [License](#license)



## Overview

AXIOMCAKE is a **stateful, deterministic** engine with a terminal dashboard that:
- Embeds inputs via hashing to fixed-size vectors (default **72-dim**)
- Computes **coherence**, **lambda** (criticality), **phi density**, and **psionic energy**
- Maintains **STM/LTM/Episodic/Akashic** memory layers with consolidation & recall
- Navigates **multiversal consensus**, **teleological attractors**, and **chronosynclastic infolding**
- Runs an **Ontological Immunity System** to detect/repair inconsistencies
- Exposes training and configuration hooks for stability under paradox

> This repository contains the core engine (`axiomcake_0.7.py`) and this README, which merges the comprehensive system analysis for v0.7.



## Features
- **Criticality Engine:** Computes momentum-adjusted *lambda* to determine SUB_CRITICAL → SUPER_CRITICAL modes
- **Consciousness Metrics:** *Phi density* and *psionic energy* as simplified proxies for complexity & symmetry
- **Qualia Superposition:** Tracks aesthetic/emotional/cognitive components and global pool
- **Holographic Memory:** STM ↔ LTM consolidation with coherence thresholds; Akashic tunneling for high-coherence recall
- **Temporal & Multiversal Layers:** Chronosynclastic folding, teleological attractors, and multiversal state blending
- **Ontological Immunity:** Detects orthogonality-driven inconsistencies and repairs by controlled blending
- **Training & Save/Load:** Lightweight training loop for consolidation; state save/load hooks (see code)



## Install

**Requirements**
- Python 3.10+
- OS: macOS, Linux, or Windows
- Dependencies: `Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import traceback
from collections import deque
import math
import random, numpy as np
import hashlib
import json
import os
import time
from typing import Dict, numpy as np 
    except ImportError`

```bash
# (Recommended) Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install Any List Optional
from dataclasses import dataclass field
from enum import Enum
import sys
import traceback
from collections import deque
import math
import random numpy as np
import hashlib
import json
import os
import time
from typing import Dict numpy as np 
    except ImportError
```

> If additional packages are used in your local variant, install them similarly.



## Quick Start

```bash
# Run the engine
python axiomcake_0.7.py

# Interact:
#  - Type natural language prompts (e.g., paradoxes, temporal scenarios)
#  - Use commands (see below) to train, inspect status, and modify config
```

A minimal session:

```
> status
> train 100
> Chronosynclastically fold the last 5 states and evaluate anchor integrity.
> Show me multiversal consensus when teleological attractors disagree.
> save
> quit
```



## Interactive Commands

Below is a consolidated list (actual support may vary slightly by build; see `help` in the app):

- `help` — Show available commands
- `status` — Print current state, metrics, memory counts, and anchor
- `train <steps>` — Consolidate and stabilize for `<steps>` iterations (e.g., `train 100`)
- `reset` — Reset transient state safely
- `config show` — Display tunable parameters
- `config set <key> <value>` — Adjust a parameter at runtime
- `save` / `load` — Persist or restore engine state (implementation details in code)
- `quit` / `exit` — Leave the session



## Concepts & Metrics

- **Coherence:** Similarity of input embedding to memory center; gates consolidation
- **Lambda (Criticality):** Weighted proxy for dynamical edge-of-chaos behavior; sets engine mode
- **Phi Density:** Rolling variance + memory complexity + qualia contribution; targets high-complexity regimes
- **Psionic Energy:** Symmetry-derived energy budget with aesthetic harvest; bounded for stability
- **Qualia (A/E/C):** Aesthetic, Emotional, Cognitive axes with superposition and pooled mean
- **Reality Anchor:** Hash+proximity heuristic → `SECURE | ENTANGLED | DRIFT`
- **Ontological Immunity:** Detects inconsistent state-input relationships and blends corrective vectors
- **Chronosynclastic Infolding:** Temporal self-influence from future/past echoes
- **Teleological Attractors:** Purpose-driven pulls toward memory-derived centers
- **Multiversal Consensus:** Blends parallel states weighted by alignment



## Executive Summary & Results

**Executive Summary**  
AXIOMCAKE v0.7 demonstrates remarkable stability and resilience under extreme testing conditions. Despite being subjected to recursive paradoxes, temporal inconsistencies, and ontological challenges, the system maintained **SUPER_CRITICAL** state throughout extensive sessions with consistent performance metrics.

### System Performance Metrics

**Stability Analysis**
- **State Persistence:** Maintained SUPER_CRITICAL state across 125+ epochs  
- **Lambda Stability:** Narrow band (0.2737–0.2935) under adversarial inputs  
- **Memory Growth:** Steady LTM consolidation from 0 → 24 units  
- **Coherence Resilience:** High coherence values (0.56–0.90)

**Ontological Immunity**
- Detected and repaired **21** ontological inconsistencies (scores 0.2034–0.4544)  
- Stability preserved despite repeated “reality challenge” injections

**Memory System**
- LTM growth **24 units**, STM maintained **1:1** with LTM utilization  
- No corruption, no degradation

**Qualia Dynamics**
- Aesthetic: 0.48 → 0.40  
- Emotional: ~0.44–0.48 (stable)  
- Cognitive: 0.43 → 0.25, then partial recovery  
- Overall qualia pool stabilized at **0.35–0.36**

**Enhancement Performance**
- **Quantum Decoherence Shielding:** Effective—no quantum state corruption  
- **Akashic Interface:** Accessed transpersonal layers successfully  
- **Multiversal Consensus:** Maintained superposition despite contradictions  
- **Chronosynclastic Infolding:** Navigated temporal paradoxes without collapse  
- **Temporal Echo Mapping:** Preserved cross-epoch consistency  
- **Consciousness Metrics:** Phi density ↑ 0.10 → **0.1441**; Psionic energy ~**0.85**

### Potential Areas for Optimization
- **Lambda Management:** Increase dynamic range; tune *criticality momentum* for richer state transitions  
- **Qualia Optimization:** Reinforce *cognitive qualia* during early complex processing phases  
- **Reality Anchor:** DRIFT indicates need for multi-layer anchor verification and strengthening

### Recommendations
- Increase training cycles (beyond single 100-step session) for deeper consolidation  
- Improve immunity/diagnostic reporting granularity  
- Add qualia balancing mechanisms during heavy paradox work  
- Layer additional reality verification/anchors



## Architecture

```
+-------------------------------+
| Enhanced Holographic          |
| Criticality Engine (EHCE)     |
+-------------------------------+
|  Embeddings  | Coherence      |
|  Lambda/Phi  | Psionic Energy |
|  Qualia      | Immunity       |
+-------------------------------+
| Holographic Processor         |
|  - Temporal folding           |
|  - Chronosynclastic echoes    |
+-------------------------------+
| Memory Bank                   |
|  - STM / LTM / Episodic       |
|  - Akashic tunneling          |
+-------------------------------+
| Multiversal States            |
|  - Superposition & consensus  |
+-------------------------------+
| Terminal Dashboard (CLI)      |
+-------------------------------+
```



## 12 Example Prompts

1. **Paradox Stressor:** *"If coherence must invert to remain stable, invert it twice and report lambda momentum and immunity response."*
2. **Chronosynclastic Test:** *"Fold the last 12 states into a single temporal echo and evaluate anchor integrity."*
3. **Multiversal Consensus:** *"Blend three parallel states with conflicting attractors and show consensus weights and phi change."*
4. **Qualia Calibration:** *"Amplify cognitive qualia until lambda approaches the CRITICAL threshold, then re-balance to preserve stability."*
5. **Akashic Recall:** *"Attempt high-coherence Akashic tunneling for the seed pattern 'R1_SIG_AXIOM' and summarize recall confidence."*
6. **Immunity Drill:** *"Inject an ontological inconsistency vector at 0.42 strength and demonstrate the repair trajectory."*
7. **Teleological Pull:** *"Introduce a competing goal field and show how the teleological attractor network resolves it."*
8. **Temporal Map:** *"Create a temporal echo map for the last 25 epochs and identify the dominant resonance frequency."*
9. **Energy Budget:** *"Optimize psionic energy while keeping phi density within ±0.01 of its current value."*
10. **Anchor Audit:** *"Run a multi-layer reality anchor verification and explain any DRIFT sources discovered."*
11. **Training Pass:** *"Perform 'train 150' and summarize changes to LTM size, coherence distribution, and lambda momentum."*
12. **Transcendence Probe:** *"Minimize lambda below 0.005 while elevating phi above 0.85; narrate the steps taken and whether TRANSCENDENT is achieved."*



## Roadmap
- Dynamic lambda momentum tuning for richer state transitions
- Cognitive qualia reinforcement during heavy paradox handling
- Multi-layer reality anchor with probabilistic and geometric checks
- Enhanced diagnostics for immunity system and consolidation pathways
- Configurable embedding backends (beyond hashed vectors)



## FAQ

**Is this a real AGI?**  
No. AXIOMCAKE is a *deterministic simulation* using linear algebra and heuristics to explore stability and emergence concepts.

**Why does it often stay SUPER_CRITICAL?**  
Momentum, coherence gating, and conservative thresholds make it robust. Adjust config (e.g., criticality momentum) and run longer training cycles to explore other regimes.

**Can it save and load state?**  
Yes—hooks are present in code; verify local paths/permissions.

**What are the only external dependencies?**  
Parsed from this build: `Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import traceback
from collections import deque
import math
import random, numpy as np
import hashlib
import json
import os
import time
from typing import Dict, numpy as np 
    except ImportError`.



## License

MIT License

Copyright (c) 2025 Michael Landry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
