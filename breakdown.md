FINAL(# Recursive Language Models: A Technical Report

## 1. Title, Authors, and Thesis
**Title**: Recursive Language Models (RLM) — Technical Report (as inferred from the paper content)

**Authors**: Not explicitly stated in the provided sections (the paper appears to be an anonymous technical report or preprint)

**One-sentence thesis**: This paper introduces Recursive Language Models (RLMs), a framework that enables LLMs to recursively call sub-LLMs within a REPL (Read-Eval-Print Loop) environment to decompose and solve complex, long-context tasks that exceed the capabilities of standard single-pass inference.

---

## 2. The Problem
The paper addresses the fundamental limitation of standard LLMs in handling tasks that require:
- **Long-context reasoning** (up to 1M+ tokens)
- **Multi-step decomposition** of complex queries
- **Verification and iterative refinement** of answers
- **Programmatic manipulation** of large contexts (codebases, datasets)

**Why it matters**: As LLMs are deployed on increasingly complex real-world tasks (codebase analysis, multi-hop retrieval, long-document reasoning), single-pass inference proves insufficient. The paper proposes that enabling models to *recursively query themselves* within a code-execution environment can overcome these limitations without requiring architectural changes or task-specific fine-tuning.

---

## 3. Key Claims / Contributions
- **RLM framework**: A task-agnostic prompting approach that gives LLMs access to a REPL environment with recursive `llm_query` and `rlm_query` functions
- **Recursive decomposition**: Models can break down complex tasks into sub-tasks, solve them hierarchically, and aggregate results
- **Model-agnostic improvements**: RLMs improve performance across multiple model families (GPT-5, Qwen3-Coder) and benchmarks
- **Cost-effectiveness**: RLMs achieve comparable or better performance at similar or lower median costs, with the ability to use cheaper sub-models for summarization
- **Scaling analysis**: Systematic evaluation of how RLM performance scales with depth and model capability
- **Failure mode documentation**: Candid reporting of what doesn't work, including model-specific limitations and prompt brittleness

---

## 4. Methodology

### 4.1 Core Architecture
The RLM system uses a **REPL (Read-Eval-Print Loop)** environment where the model can:
- Execute Python code (regex searches, data manipulation, buffer management)
- Call `llm_query()` for simple sub-tasks (summarization, extraction, direct questions)
- Call `rlm_query()` for complex sub-tasks requiring multi-step reasoning (depth > 1)
- Maintain state through variables and buffers across turns

### 4.2 System Prompt Design
The system prompt is **task-agnostic** with minor model-specific variations:

**Base prompt (GPT-5, depth=1)** includes:
- REPL environment description with `context` variable
- `llm_query` function specification
- Example strategies: chunking, iterative analysis, aggregation
- Guidance on using print statements and maintaining state

**Model-specific modifications**:
- **Qwen3-Coder-480B-A35B**: Added warning to batch `llm_query` calls (~200k chars per call) to minimize runtime
- **Qwen3-8B**: Adjusted for 32k context window (vs. 272k), reduced document batching (2-3 docs per query), smaller chunk sizes (1000 chars vs. 10000)
- **Depth > 1**: Added `rlm_query` function for recursive reasoning

### 4.3 Orchestration Framework (for decomposition hints)
The paper describes a multi-step orchestration framework:
1. **Decomposition**: Problem broken into "nodes" (self-contained subquestions) with dependencies, forming a DAG
2. **Layer-by-layer solving**: Nodes solved in parallel via `llm_batch` once dependencies verified
3. **Verification**: Every answer verified (independent second opinions or plausibility checks) before propagation
4. **Cycle handling**: Seed node guessed, downstream solved, candidates iteratively proposed until constraints satisfied
5. **Memoization**: All verified answers stored in dict, reused across turns
6. **Final assembly**: Built purely by dict lookup, never recomputation

**Core principle**: "Orchestrate, don't solve" — the main agent coordinates LLM sub-agents without performing mathematical reasoning itself.

---

## 5. Experimental Setup

### 5.1 Datasets
- **OOLONG**: Long-input benchmark with semantic category mapping tasks (1024 to 1,048,576 token contexts)
- **OOLONG-Pairs**: 20 synthetically generated tasks requiring pair identification (quadratic scaling property — cannot be solved linearly)
- **CodeQA**: ~900k token codebase analysis tasks (text-to-image model training code)
- **BrowseComp+**: Multi-hop retrieval tasks requiring document search

### 5.2 Models Evaluated
- **GPT-5** (and GPT-5.2 for LongCoT-mini experiments)
- **Qwen3-Coder-480B-A35B**
- **Qwen3-8B** (fine-tuned variant)
- **GPT-5-nano** (for summarization baseline)

### 5.3 Baselines
- Standard single-pass LLM inference
- CodeAct agent (with and without BM25 retriever)
- Summary agent (GPT-5-nano for context compression)
- LongCoT-mini (Prime Intellect's rlm-harness implementation)

### 5.4 Metrics
- Task success rate (percentage of correctly solved tasks)
- API cost (USD)
- Runtime (seconds)
- Sub-call counts (number of recursive LLM queries)

### 5.5 Hyperparameters
- RLM depth: 1 (primary), >1 (scaling experiments)
- Context window: 272k tokens (GPT-5, Qwen3-Coder), 32k (Qwen3-8B)
- Sub-LLM: GPT-5-nano for summarization tasks

---

## 6. Results

### 6.1 Main Performance Results
- **RLM(depth=1) vs. baselines**: RLMs solve equal or more tasks across benchmarks, particularly for GPT-5
- **Fine-grained comparison**: Task-level visualization shows RLMs consistently outperform or match baselines

### 6.2 Cost and Runtime Analysis
- **Median costs**: RLMs show comparable or lower costs at the 50th percentile
- **Tail behavior**: Sharp cost increases at 95th percentile due to long RLM trajectories
- **Runtime bottleneck**: Sequential (non-asynchronous) LM calls make RLMs significantly slower than baselines
- **Cost distribution**: Long-tailed, high-variance distributions for RLM methods

### 6.3 Sub-call Behavior
- **GPT-5**: Uses more sub-calls for BrowseComp-Plus (~10 per task)
- **Qwen3-Coder**: Uses ~500 sub-calls for OOLONG correct rollouts; hundreds to thousands for simple tasks
- **CodeQA-Query_212**: Qwen3-Coder made thousands of recursive sub-calls (one per line for classification)

### 6.4 Scaling Experiments
- **GPT-5.2 base vs. decomposition hints**: Base performs better overall (38.7% vs. 28.6%)
- **MATH improvement**: Hints help on MATH (37.0% vs. 26.0%)
- **LOGIC degradation**: Hints hurt LOGIC (19.1% vs. 53.6%)

### 6.5 Summary Agent Baseline
- **Cost reduction**: GPT-5-nano summarization is nearly 15x cheaper than Qwen3-Coder on BrowseComp-Plus
- **Performance**: Comparable to using full GPT-5 on a 20-sample test

### 6.6 Example Trajectories
- **Successful (BCP-74)**: RLM correctly answers via recursive verification, returning "Maria Dalmacio"
- **Failed (OOLONG-Pairs-Query_3)**: Model computes correct answer programmatically but re-verifies 5 times and returns wrong answer from root LM instead of trusting its computation
- **CodeQA-Query_44**: GPT-5 breaks down ~900k token codebase, sub-queries for clues, aggregates findings

---

## 7. Limitations and What Did NOT Work

### 7.1 System Prompt Sensitivity
- Identical prompts across models (GPT-5 vs. Qwen3-Coder) led to undesirable behavioral differences
- Required model-specific prompt adjustments

### 7.2 Model Capability Requirements
- **Qwen3-8B**: Lacked sufficient coding abilities to function effectively as RLM in REPL environment
- **Qwen3-235B**: Output token limitations — trajectories ran out of tokens mid-generation

### 7.3 Output Format Brittleness
- FINAL() tag mechanism for distinguishing answers from thoughts proved fragile
- Models occasionally output plans as final answers

### 7.4 Performance Bottlenecks
- Sequential LM calls made RLM experiments significantly slower than baselines
- 95th percentile runtimes dominated by sequential sub-LLM calls

### 7.5 Trust and Verification Failures
- **OOLONG-Pairs failure mode**: Model didn't trust its own successful computation, repeatedly re-verifying and ultimately returning wrong answer
- **Qwen3-Coder over-verification**: Frequently over-verifies answers, sometimes failing to return correct programmatically-computed answers

### 7.6 Decomposition Limitations
- Even with hints, GPT-5 cannot effectively decompose and solve sub-problems via standard chain-of-thought
- Direct chain-of-thought solving scores ~0% on complex tasks without REPL mechanism
- Decomposition hints help MATH but confuse model on programmatic tasks

### 7.7 Sub-call Inefficiency
- Qwen3-Coder makes hundreds to thousands of recursive sub-calls for simple tasks
- Could solve many tasks with fewer sub-calls

---

## 8. Takeaways

### Why This Matters
1. **Enables long-context reasoning**: RLMs successfully handle 1M+ token contexts through recursive decomposition
2. **Task-agnostic approach**: Single prompting framework works across diverse benchmarks without task-specific tuning
3. **Model-agnostic improvements**: Benefits apply across different model families and sizes
4. **Cost-effective scaling**: Cheaper sub-models (GPT-5-nano) can handle summarization, reducing overall costs ~15x
5. **Transparent failure analysis**: Candid documentation of failures provides roadmap for improvement

### What It Enables
- **Codebase analysis**: Partitioning and recursive sub-querying of ~900k token repositories
- **Multi-hop retrieval**: Complex BrowseComp+ queries solved through iterative verification
- **Hierarchical reasoning**: Models can decompose problems into DAGs of sub-problems with verification
- **Future improvements**: Asynchronous calls and better prompting strategies could address runtime bottlenecks

### Key Design Principles
1. **Orchestration over solving**: Main agent coordinates sub-agents rather than solving directly
2. **Verification before propagation**: Unverified answers never used downstream
3. **Memoization**: Verified answers reused, never recomputed
4. **Model-specific adaptation**: Prompts must be tuned to model capabilities and context windows

The paper demonstrates that recursive self-querying within a code-execution environment is a viable and powerful paradigm for extending LLM capabilities, while honestly documenting the significant engineering challenges that remain.