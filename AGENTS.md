## Charlie - The Journaling Mind Mapper

Charlie is a journaling tool that ingests unstructured journal entries, and uses `graphiti` to ingest entries as "episodes", which builds a graph database using a local Kuzu DB. Once the knowledge graph (KG) has been constructued, Charlie dynamically generates HTML pages that allow the user to traverse their entries and the KG data in a web-like manner, following links, hovering over tooltips, etc. but has properties from the following systems:

### Semantic web

MUST READS:

https://en.wikipedia.org/wiki/Semantic_Web
https://en.wikipedia.org/wiki/RDFa

Charlie follows semantic web best principles to enrich data documents with meaning, although the exact structure is yet to be determined. Whether using RDF or XML needs evaluation and an ongoing area of research.


### Project Xanadu

MUST READ: https://en.wikipedia.org/wiki/Project_Xanadu

Charlie's UI architecture follows Project Xanadu's guidelines to always show "transcluded" documents together, source and reference, so the user can follow thoughts & rerefences through time and context.
