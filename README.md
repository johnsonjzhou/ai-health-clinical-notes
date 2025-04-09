# AI tools for healthcare clinical notes

This repository contains several examples demonstrating the use of off-the-shelf AI tools for healthcare clinical notes.  

Although we make use of large language models (LLM), all of these examples can be run locally on your computer using open-source tools and models.  

**Important**: The examples shown are provided for educational purposes only and is not intended to be used as a final working product. Use these at your own risk.  

In the `examples` folder, we cover:
1. Converting patient-doctor dialogues into a structured medical report.  
2. Extract medications from patient-doctor dialogues.  
3. How to work with standardised clinical terminology like SNOMED CT-AU (with Australian Medicines Terminology).  
4. Extract medications from patient-doctor dialogues and matching them to SNOMED CT-AU concept codes.  

We will use the following tools:  
- Python Data Analysis Library (Pandas) ([link](https://pandas.pydata.org/))  
- LM Studio Python SDK ([link](https://lmstudio.ai/docs/python)). Note that the LM Studio desktop application must be running and the server is enabled within the application.  

Data used:  
- Clinical visit note summarisation corpus ([link](https://github.com/microsoft/clinical_visit_note_summarization_corpus))  
- SNOMED CT-AU with AMT ([link](https://www.digitalhealth.gov.au/healthcare-providers/product-releases/snomed-ct-au-with-australian-medicines-terminology-amt-march-2025-release))  

New to SNOMED CT-AU? Check out CSIRO's Shrimp Browser ([link](https://ontoserver.csiro.au/shrimp/?concept=138875005&valueset=http://snomed.info/sct?fhir_vs&fhir=https://tx.ontoserver.csiro.au/fhir))  

Requirements:  
- Install Python and LM Studio ([link](https://lmstudio.ai/)) on your computer.  
- Install the required Python packages listed in `requirements.txt`.  
- Download a supported LLM in LM Studio that your computer can support.  
