# Evaluation Results

## Q1: How many essential characteristics, service models, and deployment models does the NIST definition of cloud computing have?

**Answer:** The NIST definition of cloud computing is composed of five essential characteristics, three service models, and four deployment models.

**Sources (as cited by the model):**
- `docs\01_nist_800-145_cloud_computing.pdf` — chunks: 01_nist_800-145_cloud_computing::chunk::5

_gate=pass | chunks_retrieved=4 | top_score=0.783 | tokens=2086 | cost=$0.00000 | latency=1453ms_

## Q2: What is VAERS, and how do you report a reaction to it?

**Answer:** VAERS stands for the Vaccine Adverse Event Reporting System. It is a system used for reporting adverse reactions to vaccines. Your health care provider will usually file this report, or you can do it yourself by visiting the VAERS website at www.vaers.hhs.gov or calling 1-800-822-7967.

**Sources (as cited by the model):**
- `docs\03_cdc_flu_vaccine_vis.pdf` — chunks: 03_cdc_flu_vaccine_vis::chunk::1

_gate=pass | chunks_retrieved=4 | top_score=0.677 | tokens=2240 | cost=$0.00000 | latency=1172ms_

## Q3: Is the moon and distance data in this document current, and how should an answer handle that?

**Answer:** The moon and distance data provided in the document is current as of July 2013. The document explicitly notes that moon counts can change due to frequent new discoveries and provides specific notes regarding moons awaiting official confirmation as of that date. When using this data, it should be acknowledged that it reflects the status as of July 2013.

**Sources (as cited by the model):**
- `docs\02_nasa_our_solar_system.pdf` — chunks: 02_nasa_our_solar_system::chunk::2, 02_nasa_our_solar_system::chunk::3

_gate=pass | chunks_retrieved=4 | top_score=0.689 | tokens=2870 | cost=$0.00000 | latency=1219ms_

## Q4: What is the maximum towing capacity of a diesel pickup truck?

**Answer:** I don't have enough information in the provided documents to answer that question.

**Sources:** none

_gate=below_threshold | chunks_retrieved=4 | top_score=0.524 | tokens=0 | cost=$0.00000 | latency=297ms_

## Q5: How many moons does the cloud have, according to these documents?

**Answer:** The provided documents do not contain information about a "cloud" having moons. The documents discuss the Oort Cloud, which is described as a vast spherical shell of icy bodies surrounding the solar system, but they do not mention it having moons.

**Sources (as cited by the model):**
- `docs\02_nasa_our_solar_system.pdf` — chunks: 02_nasa_our_solar_system::chunk::3

_gate=pass | chunks_retrieved=4 | top_score=0.690 | tokens=2329 | cost=$0.00000 | latency=1094ms_

## Session summary

- **query_count**: 5
- **gate_passed**: 4/5
- **gate_no_context**: 1/5
- **no_context_rate**: 0.2
- **empty_index_events**: 0
- **below_threshold_events**: 1
- **retrieval_failures**: 0
- **structured_output_failures**: 0
- **avg_total_latency_ms**: 1047.0
- **avg_retriever_latency_ms**: 334.6
- **avg_top_similarity_score**: 0.672
- **avg_similarity_score_all_chunks**: 0.633
- **total_estimated_cost_usd**: 0.0
- **total_tokens**: 9525
- **total_chunks_dropped_for_budget**: 0
- **total_neighbor_chunks_added**: 0