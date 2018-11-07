# IUST Test Corpus

We are gathering some large corpus for different file formats such as portable document format (PDF), extensible markup language (XML), and hypertext markup language (HTML) to do fuzz testing real word application which take these formats as input their majoring inputs. At this time, IUST PDF Corpus is ready to view and download. 

## IUST PDF Corpus

![IUSTPDFCorpusDemo Image](IUSTPDFCorpusDemo.PNG)

We are happy to introduce **IUST PDF Corpus**, a large set of various PDF files, aimed at testing and improving the qualification of real word PDF readers such as [MuPDF](https://mupdf.com/).
IUST PDF Corpus contains about **6,000** PDF file. we extract more than **500,000** PDF data object from this corpus to evaluate IUST DeepFuzz.  

The extracted objects have put under a _pdfs_ directory. We divide the objects dataset into two sub-dataset: large-size and small-size. The small-size dataset is created to develop and test the generative models and has about 120,000 PDF objects. The large dataset is used to train deep models and fuzz testing PDF viewers and has 500,000 PDF objects.
We are extending this corpus and want to add more PDF files, as soon as possible.
We also extract 1000 binary streams form data objects. These streams have put under the small-size subdirectory. All extracted objects are available to view and download from the current GitHub repository. The complete set of PDF files will be availble to view and download after our paper on IUST DeepFuzz publish. 

[View and download IUST PDF Corpus (version 1.0)](#)


## IUST XML Corpus

Comming soon.


## IUST HTML Corpus

Comming soon.

