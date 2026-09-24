

# CFRS

**Confidence-Focused Multimodal Sentiment Analysis via Reliability Supervision (CFRS)**.

## Abstract

Multimodal Sentiment Analysis (MSA) aims to infer human sentiment by integrating visual, acoustic, and textual information. Many MSA methods consider how to reduce cross-modal distribution gaps, and some methods also consider the reliability of modality representations during multimodal fusion. In this paper, we propose **Confidence-Focused Multimodal Sentiment Analysis via Reliability Supervision (CFRS)**, which mainly comprises the **Common Representation Learning (CRL)** module and the **Confidence Modeling (CM)** module. In the CRL module, we employ normalizing flows to align the common representations across modalities. The CM module consists of a **Confidence Estimation (CE)** block and a **Reliability Supervision (RS)** block, where the RS block is designed to calibrate the confidence scores, enabling reliable weighted fusion of modality-common and modality-specific representations. Experimental results demonstrate that our proposed CFRS achieves the best performance on Acc-2, Acc-7, and F1 on both the CMU-MOSEI and CMU-MOSI datasets compared with ten representative MSA methods.

## Code Release



