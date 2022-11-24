# X-ray Fingerprints

A project exploring how uniquely identifying machine traits are in x-rays and how to properly anonymize x-rays

This work will need to show many things:
* how uniquely these fingerprints identify things (manufacturer level or individual machine level)
* that even without using ML there is a clear signature
* that common simple transformations that most people are using do not eliminate these signatures
* that a DL algorithm can indeed quickly learn to identify the site or machine from an image
* then show what transformations are necessary so that the prior DL algorithm can no longer determine machine ID or site ID
* a list of commonly studied OAI variables that have been used by OAI-DL projects that are not evenly distributed across machines or sites (thus giving algorithms opportunity to cheat)
