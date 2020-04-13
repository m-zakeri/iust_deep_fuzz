# IUST Neural Software Testing (IUST-NST) Dataset

Neural software testing (NST) is about applying machine learning techniques, specially deep-learning and neural network, in the field of software testing. We began with fuzz testing, but it can transform into other types of software testing. An unavoidable part of all machine learning task is data. The goal of this section is to provide suitable and public dataset which can be used by other researchers.

For now, we are gathering some large corpus for different file formats such as portable document format (PDF), extensible markup language (XML), and hypertext markup language (HTML) to do fuzz testing real-world application which takes these formats as their majoring inputs.
At this time, IUST PDF Corpus is ready to view and download. 
We refer the user on [dataset subdirectory](../dataset/) of project which is available  on Github

### News
**2019-10-13:** IUST-PDFCorpus version 1.0.0 is publicly available at [https://zenodo.org/record/3484013](https://zenodo.org/record/3484013) with DOI **10.5281/zenodo.3484013**.


## IUST-PDFCorpus
**Download:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3484013.svg)](https://doi.org/10.5281/zenodo.3484013)

 ![IUSTPDFCorpusDemo Image](pdfs/IUST-PDFCorpusDemo.PNG)

We are happy to introduce **IUST PDF Corpus**, a large set of various PDF files, aimed at manipulating, 
testing and improving the qualification of real-world PDF readers such as [MuPDF](https://mupdf.com/).
IUST PDF Corpus (version 1.0) contains **6,141** PDF file. we extract more than **500,000** PDF data object from this corpus to evaluate IUST DeepFuzz, our new file format fuzzer. 

The extracted objects have put under a _pdfs_ directory. We divide the objects dataset into two sub-dataset: _large-size_ and _small-size_. The small-size dataset is created to develop and test the generative models and has about 120,000 PDF objects. The large dataset is used to train deep models and fuzz testing PDF viewers and has 500,000 PDF objects.
We are extending this corpus and want to add more PDF files, as soon as possible.
We also extract 1000 binary streams form data objects. These streams have put under the small-size subdirectory. All extracted objects are available to [view and download](./pdfs/) from the current GitHub repository. The complete set of PDF files will be available to view and download as soon as our relevant paper on IUST DeepFuzz is published.


## IUST XML Corpus

Under gathering, construction and sanitizing. Coming soon :)


## IUST HTML Corpus

Under gathering, construction and sanitizing. Coming Soon :)

