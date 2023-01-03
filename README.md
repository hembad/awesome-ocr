# Awesome OCR

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This list contains links to great software tools and libraries and literature
related to [Optical Character Recognition
(OCR)](http://en.wikipedia.org/wiki/Optical_Character_Recognition).

Contributions are welcome, as is feedback.

<!-- BEGIN-MARKDOWN-TOC -->

- [Awesome OCR](#awesome-ocr)
  - [Software](#software)
    - [OCR engines](#ocr-engines)
    - [Older and possibly abandoned OCR engines](#older-and-possibly-abandoned-ocr-engines)
    - [OCR file formats](#ocr-file-formats)
      - [hOCR](#hocr)
      - [ALTO XML](#alto-xml)
      - [TEI](#tei)
      - [PAGE XML](#page-xml)
    - [OCR CLI](#ocr-cli)
  - [Deskewing and Dewarping](#deskewing-and-dewarping)
    - [OCR GUI](#ocr-gui)
  - [Text detection and localization](#text-detection-and-localization)
    - [OCR Preprocessing](#ocr-preprocessing)
  - [Segmentation](#segmentation)
    - [Line Segmentation](#line-segmentation)
    - [Character Segmentation](#character-segmentation)
    - [Word Segmentation](#word-segmentation)
    - [Document Segmentation](#document-segmentation)
    - [Form Segmentation](#form-segmentation)
  - [Handwritten](#handwritten)
  - [Table detection](#table-detection)
  - [Language detection](#language-detection)
    - [OCR as a Service](#ocr-as-a-service)
    - [OCR evaluation](#ocr-evaluation)
    - [OCR libraries by programming language](#ocr-libraries-by-programming-language)
      - [Crystal](#crystal)
      - [Elixir](#elixir)
      - [Go](#go)
      - [Java](#java)
      - [.Net](#net)
      - [Object Pascal](#object-pascal)
      - [PHP](#php)
      - [Python](#python)
      - [Javascript](#javascript)
      - [Ruby](#ruby)
      - [Rust](#rust)
      - [R](#r)
      - [Swift](#swift)
    - [OCR training tools](#ocr-training-tools)
  - [Datasets](#datasets)
    - [Ground Truth](#ground-truth)
  - [Video Text Spotting](#video-text-spotting)
  - [Font detection](#font-detection)
  - [Optical Character Recognition Engines and Frameworks](#optical-character-recognition-engines-and-frameworks)
  - [Awesome lists](#awesome-lists)
  - [Proprietary OCR Engines](#proprietary-ocr-engines)
  - [Cloud based OCR Engines (SaaS)](#cloud-based-ocr-engines-saas)
  - [File formats and tools](#file-formats-and-tools)
  - [Datasets](#datasets-1)
  - [Data augmentation and Synthetic data generation](#data-augmentation-and-synthetic-data-generation)
  - [Pre OCR Processing](#pre-ocr-processing)
  - [Post OCR Correction](#post-ocr-correction)
  - [Benchmarks](#benchmarks)
  - [misc](#misc)
  - [Literature](#literature)
    - [OCR-related publication and link lists](#ocr-related-publication-and-link-lists)
    - [Blog Posts and Tutorials](#blog-posts-and-tutorials)
    - [OCR Showcases](#ocr-showcases)
    - [Academic articles](#academic-articles)
      - [2011 and before](#2011-and-before)
      - [2012](#2012)
      - [2013](#2013)
      - [2014](#2014)
      - [2015](#2015)
      - [2016](#2016)
      - [2017](#2017)
      - [2018](#2018)
      - [2019](#2019)
      - [2020](#2020)

<!-- END-MARKDOWN-TOC -->

## Software

### OCR engines

- [tesseract](https://github.com/tesseract-ocr/tesseract) - The definitive Open Source OCR engine `Apache 2.0`
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - OCR engine built on PyTorch by JaidedAI, `Apache 2.0`
- [ocropus](https://github.com/tmbdev/ocropy) - OCR engine based on LSTM, `Apache 2.0`
- [ocropus 0.4](https://github.com/jkrall/ocropus) - Older v0.4 state of Ocropus, with tesseract 2.04 and iulib, C++
- [kraken](https://github.com/mittagessen/kraken) - Ocropus fork with sane defaults
- [gocr](https://www-e.ovgu.de/jschulen/ocr/) - OCR engine under the GNU Public License led by Joerg Schulenburg.
- [Ocrad](http://www.gnu.org/software/ocrad/) - The GNU OCR. `GPL`
- [ocular](https://github.com/tberg12/ocular) - Machine-learning OCR for historic documents
- [SwiftOCR](https://github.com/garnele007/SwiftOCR) - fast and simple OCR library written in Swift
- [attention-ocr](https://github.com/emedvedev/attention-ocr) - OCR engine using visual attention mechanisms
- [RWTH-OCR](https://www-i6.informatik.rwth-aachen.de/rwth-ocr/) - The RWTH Aachen University Optical Character Recognition System
- [simple-ocr-opencv](https://github.com/goncalopp/simple-ocr-opencv) and its [fork](https://github.com/RedFantom/simple-ocr-opencv) - A simple pythonic OCR engine using opencv and numpy
- [Calamari](https://github.com/Calamari-OCR/calamari) - OCR Engine based on OCRopy and Kraken
- [doctr](https://github.com/mindee/doctr) - A seamless & high-performing OCR library powered by Deep Learning

### Older and possibly abandoned OCR engines

- [Clara OCR](http://freecode.com/projects/claraocr/) - Open source OCR in C `GPL`
- [Cuneiform](<https://en.wikipedia.org/wiki/CuneiForm_(software)>) - CuneiForm OCR was developed by Cognitive Technologies
- [Eye](https://sourceforge.net/projects/eyeocr/) - an experimental Java OCR (image-to-text) application
- [kognition](https://sourceforge.net/projects/kognition/) - An omnifont OCR software for KDE
- [OCRchie](https://people.eecs.berkeley.edu/~fateman/kathey/ocrchie.html) - Modular Optical Character Recognition Software
- [ocre](http://lem.eui.upm.es/ocre.html) - o.c.r. easy
- [xplab](http://www.pattern-lab.de/) - A GTK 2 tool for pattern matching
- [hebOCR](https://github.com/yaacov/hebocr) - Hebrew character recognition library (previously named hocr, see [Wikipedia article](https://de.wikipedia.org/wiki/HebOCR)) `GPL`

### OCR file formats

- [abby2hocr.xslt XSLT script](https://gist.github.com/tfmorris/5977784)
- [ocr-conversion-scripts](https://github.com/cneud/ocr-conversion-scripts)

#### hOCR

- [hocr-tools](https://github.com/tmbdev/hocr-tools) - Tools for doing various useful things with hOCR files, `Apache 2.0`
- [hocr-spec](https://github.com/kba/hocr-spec) - hOCR 1.2 specification
- [ocr-transform](https://github.com/UB-Mannheim/ocr-transform) - CLI tool to convert between hOCR and ALTO, `MIT`
- [hocr-parser](https://github.com/athento/hocr-parser) - hOCR Specification Python Parser
- [hOCRTools](https://github.com/ONB-RD/hOCRTools) - hOCR to ALTO conversion XSLT

#### ALTO XML

- [ALTO XML Schema](https://github.com/altoxml/schema) - XML Schema and development of the ALTO XML format
- [ALTO XML Documentation](https://github.com/altoxml/documentation) - Documentation and use cases for ALTO
- [alto-tools](https://github.com/cneud/alto-tools) - Various tools to work with ALTO files, Python
- [AbbyyToAlto](https://github.com/ironymark/AbbyyToAlto) - PHP script converting from Abbyy 6 to ALTO XML

#### TEI

- [TEI-OCR](https://github.com/OpenPhilology/tei-ocr) - TEI customization for OCR generated layout and content information
- [TEI SIG on Libraries](http://www.tei-c.org/SIG/Libraries/teiinlibraries/main-driver.html) - Best Practices for TEI in Libraries
- [GDZ](http://gdz.sub.uni-goettingen.de/uploads/media/GDZ_document_format_2005_12_08.pdf) - METS/TEI-based GDZ document format

#### PAGE XML

- [PAGE-XML Schema](https://github.com/PRImA-Research-Lab/PAGE-XML/tree/master/pagecontent) - XML schema of the PAGE XML format along with documentation and examples
- [omni:us Pages Format (OPF)](https://omni-us.github.io/pageformat/pagecontent_omnius.html) - XML schema very similar to PAGE XML that has some additional features.
- [py-pagexml](https://github.com/omni-us/pagexml) - Python library for handling PAGE XML and OPF files.

### OCR CLI

- [OCRmyPDF](https://github.com/jbarlow83/OCRmyPDF) - OCRmyPDF adds an OCR text layer to scanned PDF files, allowing them to be searched
- [Pdf2PdfOCR](https://github.com/LeoFCardoso/pdf2pdfocr) - A tool to OCR a PDF (or supported images) and add a text "layer" (a "pdf sandwich") in the original file making it a searchable PDF. GUI included. Tesseract and cuneiform supported.
- [Ocrocis](https://github.com/kaumanns/ocrocis) - Project manager interface for Ocropy, see also [external project homepage](http://cistern.cis.lmu.de/ocrocis/)
- [tesseract-recognize](https://github.com/mauvilsa/tesseract-recognize) - Tesseract-based tool that outputs result in Page XML format ([docker image](https://hub.docker.com/r/mauvilsa/tesseract-recognize)).

## Deskewing and Dewarping

- [MORAN_v2](https://github.com/Canjie-Luo/MORAN_v2) ([paper:2019](https://arxiv.org/abs/1901.03003)) - A Multi-Object Rectified Attention Network for Scene Text Recognition
- [thomasjhaung/deep-learning-for-document-dewarping](https://github.com/thomasjhuang/deep-learning-for-document-dewarping)
- [unproject_text](https://github.com/mzucker/unproject_text) - Perspective recovery of text using transformed ellipses
- [unpaper](https://github.com/Flameeyes/unpaper) - a post-processing tool for scanned sheets of paper, especially for book pages that have been scanned from previously created photocopies.
- [deskew](https://github.com/sbrunner/deskew) - Library used to deskew a scanned document
- [deskewing](https://github.com/sauravbiswasiupr/deskewing) - Contains code to deskew images using MLPs, LSTMs and LLS tranformations
- [skew_correction](https://github.com/prajwalmylar/skew_correction) - De-skewing images with slanted content by finding the deviation using Canny Edge Detection.
- [page_dewarp](https://github.com/mzucker/page_dewarp) - Page dewarping and thresholding using a "cubic sheet" model
- [text_deskewing](https://github.com/dehaisea/text_deskewing) - Rotate text images if they are not straight for better text detection and recognition.
- [galfar/deskew](https://github.com/galfar/deskew) - Deskew is a command line tool for deskewing scanned text documents. It uses Hough transform to detect "text lines" in the image. As an output, you get an image rotated so that the lines are horizontal.
- [xellows1305/Document-Image-Dewarping](https://github.com/xellows1305/Document-Image-Dewarping) - No code :(
- https://github.com/RaymondMcGuire/BOOK-CONTENT-SEGMENTATION-AND-DEWARPING
- [Docuwarp](https://github.com/thomasjhuang/deep-learning-for-document-dewarping)
- [Alyn](https://github.com/kakul/Alyn)
- [DewarpNet](https://github.com/cvlab-stonybrook/DewarpNet)
-

### OCR GUI

- [moz-hocr-editor](https://github.com/garrison/moz-hocr-edit) - Firefox Addon for editing hOCR files **Discontinued**
- [qt-box-editor](https://github.com/zdenop/qt-box-editor) - QT4 editor of tesseract-ocr box files.
- [ocr-gt-tools](https://github.com/UB-Mannheim/ocr-gt-tools) - Client-Server application for editing OCR ground truth.
- [Paperwork](https://github.com/openpaperwork/paperwork) - Using scanners and OCR to grep paper documents the easy way.
- [Paperless](https://github.com/danielquinn/paperless) - Scan, index, and archive all of your paper documents.
- [gImageReader](https://github.com/manisandro/gImageReader) - gImageReader is a simple Gtk/Qt front-end to tesseract-ocr.
- [VietOCR](http://vietocr.sourceforge.net/) - A Java/.NET GUI frontend for Tesseract OCR engine, including [jTessBoxEditor](http://vietocr.sourceforge.net/training.html) a graphical Tesseract [box data](https://github.com/tesseract-ocr/tesseract/wiki/Make-Box-Files) editor
- [PoCoTo](https://github.com/cisocrgroup/PoCoTo) - Fast interactive batch corrections of complete OCR error series in OCR'ed historical documents.
- [OCRFeeder](https://wiki.gnome.org/Apps/OCRFeeder) - GTK graphical user interface that allows the users to correct characters or bounding boxes, ODT export and more.
- [PRImA PAGE Viewer](https://github.com/PRImA-Research-Lab/prima-page-viewer) - Java based viewer for PAGE XML files (layout + text content). Also supports ALTO XML, FineReader XML, and HOCR.
- [LAREX](https://github.com/chreul/larex) - A semi-automatic open-source tool for Layout Analysis and Region EXtraction on early printed books.
- [archiscribe](https://github.com/jbaiter/archiscribe) - Web application for transcribing OCR ground truth from Archive.org. Deployed instance available at https://archiscribe.jbaiter.de/, results are available in [@jbaiter/archiscribe-corpus](https://github.com/jbaiter/archiscribe-corpus).
- [nw-page-editor](https://github.com/mauvilsa/nw-page-editor) - Simple app for visual editing of Page XML files. Provides desktop and [server docker-based](https://hub.docker.com/r/mauvilsa/nw-page-editor-web) versions.

## Text detection and localization

- [DB](https://github.com/MhLiao/DB)
- [DeepReg](https://github.com/DeepRegNet/DeepReg)
- [CornerText](https://github.com/lvpengyuan/corner) - [paper:2018](https://arxiv.org/abs/1802.08948)) - Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation
- [RRPN](https://github.com/mjq11302010044/RRPN) - ([paper:2018](https://arxiv.org/abs/1703.01086)) - Arbitrary-Oriented Scene Text Detection via Rotation Proposals
- [MASTER-TF](https://github.com/jiangxiluning/MASTER-TF) - ([paper:2021](https://arxiv.org/abs/1910.02562)) - TensorFlow reimplementation of "MASTER: Multi-Aspect Non-local Network for Scene Text Recognition" (Pattern Recognition 2021).
- [MaskTextSpotterV3](https://github.com/MhLiao/MaskTextSpotterV3) - ([paper:2020](https://arxiv.org/abs/2007.09482)) - Mask TextSpotter v3 is an end-to-end trainable scene text spotter that adopts a Segmentation Proposal Network (SPN) instead of an RPN.
- [TextFuseNet](https://github.com/ying09/TextFuseNet) - ([paper:2020](https://www.ijcai.org/Proceedings/2020/72)) A PyTorch implementation of "TextFuseNet: Scene Text Detection with Richer Fused Features".
- [SATRN](https://github.com/clovaai/SATRN)- ([paper:2020](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w34/Lee_On_Recognizing_Texts_of_Arbitrary_Shapes_With_2D_Self-Attention_CVPRW_2020_paper.pdf)) - Official Tensorflow Implementation of Self-Attention Text Recognition Network (SATRN) (CVPR Workshop WTDDLE 2020).
- [cvpr20-scatter-text-recognizer](https://github.com/phantrdat/cvpr20-scatter-text-recognizer) - ([paper:2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Litman_SCATTER_Selective_Context_Attentional_Scene_Text_Recognizer_CVPR_2020_paper.pdf)) - Unofficial implementation of CVPR 2020 paper "SCATTER: Selective Context Attentional Scene Text Recognizer"
- [seed](https://github.com/Pay20Y/SEED) - ([paper:2020[https://arxiv.org/pdf/2005.10977.pdf]) - This is the implementation of the paper "SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition"
- [vedastr](https://github.com/Media-Smart/vedastr) - A scene text recognition toolbox based on PyTorch
- [AutoSTR](https://github.com/AutoML-4Paradigm/AutoSTR) - ([paper:2020](https://arxiv.org/pdf/2003.06567.pdf)) Efficient Backbone Search for Scene Text Recognition
- [Decoupled-attention-network](https://github.com/Wang-Tianwei/Decoupled-attention-network) - ([paper:2019](https://arxiv.org/abs/1912.10205)) Pytorch implementation for "Decoupled attention network for text recognition".
- [Bi-STET](https://github.com/MauritsBleeker/Bi-STET) - ([paper:2020](https://arxiv.org/pdf/1912.03656.pdf)) Implementation of Bidirectional Scene Text Recognition with a Single Decoder
- [kiss](https://github.com/Bartzi/kiss) - ([paper:2019](https://arxiv.org/abs/1911.08400)
- [Deformable Text Recognition](https://github.com/Alpaca07/dtr) - ([paper:2019](https://ieeexplore.ieee.org/abstract/document/9064428))
- [MaskTextSpotter](https://github.com/MhLiao/MaskTextSpotter) - ([paper:2019](https://ieeexplore.ieee.org/document/8812908))
- [CUTIE](https://github.com/vsymbol/CUTIE) - ([paper:2019](https://arxiv.org/abs/1903.12363v4)
- [AttentionOCR](https://github.com/zhang0jhon/AttentionOCR) - ([paper:2019](https://arxiv.org/abs/1912.04561))
- [crpn](https://github.com/xhzdeng/crpn) - ([paper:2019](https://arxiv.org/abs/1804.02690))
- [Scene-Text-Detection-with-SPECNET](https://github.com/AirBernard/Scene-Text-Detection-with-SPCNET) - Repository for Scene Text Detection with Supervised Pyramid Context Network with tensorflow.
- [Character-Region-Awareness-for-Text-Detection](https://github.com/guruL/Character-Region-Awareness-for-Text-Detection-)
- [Real-time-Scene-Text-Detection-and-Recognition-System](https://github.com/fnzhan/Real-time-Scene-Text-Detection-and-Recognition-System) - End-to-end pipeline for real-time scene text detection and recognition.
- [ocr_attention](https://github.com/marvis/ocr_attention) - Robust Scene Text Recognition with Automatic Rectification.
- [masktextspotter.caffee2](https://github.com/lvpengyuan/masktextspotter.caffe2) - The code of "Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes".
- [InceptText-Tensorflow](https://github.com/xieyufei1993/InceptText-Tensorflow) - An Implementation of the alogrithm in paper IncepText: A New Inception-Text Module with Deformable PSROI Pooling for Multi-Oriented Scene Text Detection.
- [textspotter](https://github.com/tonghe90/textspotter) - An End-to-End TextSpotter with Explicit Alignment and Attention
- [RRD](https://github.com/MhLiao/RRD) - RRD: Rotation-Sensitive Regression for Oriented Scene Text Detection.
- [crpn](https://github.com/xhzdeng/crpn) - Corner-based Region Proposal Network.
- [SSTDNet](https://github.com/HotaekHan/SSTDNet) - Implement 'Single Shot Text Detector with Regional Attention, ICCV 2017 Spotlight'.
- [R2CNN](https://github.com/beacandler/R2CNN) - caffe re-implementation of R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection.
- [RRPN](https://github.com/mjq11302010044/RRPN) - Source code of RRPN ---- Arbitrary-Oriented Scene Text Detection via Rotation Proposals
- [Tensorflow_SceneText_Oriented_Box_Predictor](https://github.com/dafanghe/Tensorflow_SceneText_Oriented_Box_Predictor) - This project modify tensorflow object detection api code to predict oriented bounding boxes. It can be used for scene text detection.
- [DeepSceneTextReader](https://github.com/dafanghe/DeepSceneTextReader) - This is a c++ project deploying a deep scene text reading pipeline with tensorflow. It reads text from natural scene images. It uses frozen tensorflow graphs. The detector detect scene text locations. The recognizer reads word from each detected bounding box.
- [DeRPN](https://github.com/HCIILAB/DeRPN) - A novel region proposal network for more general object detection ( including scene text detection ).
- [Bartzi/see](https://github.com/Bartzi/see) - SEE: Towards Semi-Supervised End-to-End Scene Text Recognition
- [Bartzi/stn-ocr](https://github.com/Bartzi/stn-ocr) - Code for the paper STN-OCR: A single Neural Network for Text Detection and Text Recognition
- [beacandler/R2CNN](https://github.com/beacandler/R2CNN) - caffe re-implementation of R2CNN: Rotational Region CNN for Orientation Robust Scene Text Detection
- [HsiehYiChia/Scene-text-recognition](https://github.com/HsiehYiChia/Scene-text-recognition) - Scene text detection and recognition based on Extremal Region(ER)
- [R2CNN_Faster-RCNN_Tensorflow](https://github.com/DetectionTeamUCAS/R2CNN_Faster-RCNN_Tensorflow) - Rotational region detection based on Faster-RCNN.
- [corner](https://github.com/lvpengyuan/corner) - Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation
- [Corner_Segmentation_TextDetection](https://github.com/JK-Rao/Corner_Segmentation_TextDetection) - Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation.
- [TextSnake.pytorch](https://github.com/princewang1994/TextSnake.pytorch) - A PyTorch implementation of ECCV2018 Paper: TextSnake: A Flexible Representation for Detecting Text of Arbitrary Shapes
- [AON](https://github.com/huizhang0110/AON) - Implementation for CVPR 2018 text recognition Paper by Tensorflow: "AON: Towards Arbitrarily-Oriented Text Recognition"
- [pixel_link](https://github.com/ZJULearning/pixel_link) - Implementation of our paper 'PixelLink: Detecting Scene Text via Instance Segmentation' in AAAI2018
- [seglink](https://github.com/dengdan/seglink) - An Implementation of the seglink alogrithm in paper Detecting Oriented Text in Natural Images by Linking Segments (=> pixe_link)
- [SSTD](https://github.com/BestSonny/SSTD) - Single Shot Text Detector with Regional Attention
- [MORAN_v2](https://github.com/Canjie-Luo/MORAN_v2) - MORAN: A Multi-Object Rectified Attention Network for Scene Text Recognition
- [Curve-Text-Detector](https://github.com/Yuliang-Liu/Curve-Text-Detector) - This repository provides train＆test code, dataset, det.&rec. annotation, evaluation script, annotation tool, and ranking table.
- [HCIILAB/DeRPN](https://github.com/HCIILAB/DeRPN) - A novel region proposal network for more general object detection ( including scene text detection ).
- [TextField](https://github.com/YukangWang/TextField) - TextField: Learning A Deep Direction Field for Irregular Scene Text Detection (TIP 2019)
- [tensorflow-TextMountain](https://github.com/liny23/tensorflow-TextMountain) - TextMountain: Accurate Scene Text Detection via Instance Segmentation
- [Bartzi/see](https://github.com/Bartzi/see) - Code for the AAAI 2018 publication "SEE: Towards Semi-Supervised End-to-End Scene Text Recognition"
- [bgshih/aster](https://github.com/bgshih/aster) - Recognizing cropped text in natural images.
- [ReceiptParser](https://github.com/ReceiptManager/receipt-parser) - A fuzzy receipt parser written in Python.
- [vedastr](https://github.com/Media-Smart/vedastr)

### OCR Preprocessing

- [NoiseRemove.java in MathOCR](https://github.com/chungkwong/MathOCR/blob/master/src/main/java/com/github/chungkwong/mathocr/preprocess/NoiseRemove.java) - Java implementation of Adaptive degraded document image binarization by B. Gatos , I. Pratikakis, S.J. Perantonis
- [binarize.c in ZBar](https://github.com/ZBar/ZBar/blob/master/zbar/qrcode/binarize.c) - C implementations of two binarization algorithms, based on Sauvola
- [typeface-corpus](https://github.com/jbest/typeface-corpus) - A repository for typefaces to train Tesseract and OCRopus for natural history collections and digital humanities.
- [binarizewolfjolion](https://github.com/zp-j/binarizewolfjolion) - Comparison of binarization algorithms. [Blog post](http://zp-j.github.io/2013/10/04/document-binarization/)
- [`crop_morphology.py` in oldnyc](https://github.com/danvk/oldnyc) - Cropping a page to just the text block
- [Whiteboard Picture Cleaner](https://gist.github.com/lelandbatey/8677901) - Shell one-liner/script to clean up and beautify photos of whiteboards
- Fred's ImageMagick script [textcleaner](http://www.fmwconcepts.com/imagemagick/textcleaner/index.php) - Processes a scanned document of text to clean the text background
- [localcontrast](https://sourceforge.net/projects/localcontrast/) - Fast O(1) local contrast optimization

## Segmentation

### Line Segmentation

- [ARU-Net](https://github.com/TobiasGruening/ARU-Net) - Deep Learning Chinese Word Segment
- [sbb_textline_detector](https://github.com/qurator-spk/sbb_textline_detector)

### Character Segmentation

- [watersink/Character-Segmentation](https://github.com/watersink/Character-Segmentation)
- [sharatsawhney/character_segmentation](https://github.com/sharatsawhney/character_segmentation)

### Word Segmentation

- [githubharald/WordSegmentation](https://github.com/githubharald/WordSegmentation)
- [kcws](https://github.com/koth/kcws)

### Document Segmentation

- [LayoutParser](https://layout-parser.github.io)
- [eynollah](https://github.com/qurator-spk/eynollah)
- [chulwoopack/docstrum](https://github.com/chulwoopack/docstrum)
- [LAREX](https://github.com/OCR4all/LAREX) - LAREX is a semi-automatic open-source tool for layout analysis on early printed books.
- [leonlulu/DeepLayout](https://github.com/leonlulu/DeepLayout) - Deep learning based page layout analysis
- [dhSegment](https://github.com/dhlab-epfl/dhSegment)
- [Pay20Y/Layout_Analysis](https://github.com/Pay20Y/Layout_Analysis)
- [rbaguila/document-layout-analysis](https://github.com/rbaguila/document-layout-analysis)
- [P2PaLA](https://github.com/lquirosd/P2PaLA) - Page to PAGE Layout Analysis Tool
- [ocroseg](https://github.com/NVlabs/ocroseg/) - This is a deep learning model for page layout analysis / segmentation.
- [DIVA-DIA/DIVA_Layout_Analysis_Evaluator](https://github.com/DIVA-DIA/DIVA_Layout_Analysis_Evaluator) - Layout Analysis Evaluator for the ICDAR 2017 competition on Layout Analysis for Challenging Medieval Manuscripts
- [ocrsegment](https://github.com/watersink/ocrsegment) - a deep learning model for page layout analysis / segmentation.
- [ARU-Net](https://github.com/TobiasGruening/ARU-Net)
- [xy-cut-tree](https://github.com/kavishgambhir/xy-cut-tree)
- [ocrd_segment](https://github.com/OCR-D/ocrd_segment)
- [LayoutML](https://github.com/microsoft/unilm/tree/master/layoutlm)
- [LayoutLMv2](https://github.com/microsoft/unilm/tree/master/layoutlmv2)
- [eynollah](https://github.com/qurator-spk/eynollah)

### Form Segmentation

- https://github.com/doxakis/form-segmentation

## Handwritten

- https://github.com/arthurflor23/handwritten-text-recognition
- https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet
- https://github.com/0x454447415244/HandwritingRecognitionSystem
- https://github.com/SparshaSaha/Handwritten-Number-Recognition-With-Image-Segmentation
- https://github.com/ThomasDelteil/HandwrittenTextRecognition_MXNet
- [SimpleHTR](https://github.com/githubharald/SimpleHTR) - Handwritten Text Recognition (HTR) system implemented with TensorFlow.
- [handwriting-ocr](https://github.com/Breta01/handwriting-ocr) - OCR software for recognition of handwritten text
- [AWSLabs: handwritten text regognition for Apache MXNet](https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet)
- [vloison/Handwritten_Text_Recognition](https://github.com/vloison/Handwritten_Text_Recognition)
- https://github.com/sushant097/Handwritten-Line-Text-Recognition-using-Deep-Learning-with-Tensorflow
- https://github.com/qurator-spk/sbb_textline_detection

## Table detection

- [TableNet](https://github.com/jainammm/TableNet) - Unofficial implementation of ICDAR 2019 paper : TableNet: Deep Learning model for end-to-end Table detection and Tabular data extraction from Scanned Document Images.
- [image-table-ocr](https://github.com/eihli/image-table-ocr)
- [TreeStructure](https://github.com/HazyResearch/TreeStructure) - Table Extraction Tool
- [TableTrainNet](https://github.com/mawanda-jun/TableTrainNet) - Table recognition inside douments using neural networks.
- [table_layout_detection_research](https://github.com/cbgaindia/parsers/blob/master/research/layout_detection_research.md)
- [TableBank](https://github.com/doc-analysis/TableBank)
- [Camelot](https://github.com/atlanhq/camelot)
- [ocr-table](https://github.com/cseas/ocr-table) - Extract tables from scanned image PDFs using Optical Character Recognition.
- [ExtractTable-py](https://github.com/ExtractTable/ExtractTable-py)
- [image-table-ocr](https://github.com/eihli/image-table-ocr)

## Language detection

- [lingua](https://github.com/pemistahl/lingua) - The most accurate natural language detection library for Java and other JVM languages, suitable for long and short text alike
- [langdetect](https://pypi.org/project/langdetect/)
- [whatthelang](https://github.com/indix/whatthelang) - Lightning Fast Language Prediction rocket
- [wiki-lang-detect](https://github.com/vseloved/wiki-lang-detect)

### OCR as a Service

- [Open OCR](https://github.com/tleyden/open-ocr) - Run Tesseract in Docker containers
- [tesseract-web-service](https://github.com/guitarmind/tesseract-web-service) - An implementation of RESTful web service for tesseract-OCR using tornado.
- [docker-ocropy](https://github.com/kba/docker-ocropy) - A Docker container for running the [ocropy OCR system](htps://github.com/tmbdev/ocropy).
- [ABBYY Cloud OCR SDK Code samples](https://github.com/abbyysdk/ocrsdk.com) - Code samples for using the proprietary commercial ABBYY OCR API.
- [nidaba](https://github.com/OpenPhilology/nidaba) - An expandable and scalable OCR pipeline
- [gamera](https://github.com/hsnr-gamera/gamera) - A meta-framework for building document processing applications, e.g. OCR
- [ocr-tools](https://github.com/subugoe/ocr-tools) - Project to provide CLI and web service interfaces to common OCR engines
- [ocrad-docker](https://github.com/kba/ocrad-docker) - Run the [ocrad](http://www.gnu.org/software/ocrad/) OCR engine in a docker container
- [kraken-docker](https://github.com/kba/kraken-docker) - Run the [kraken](https://github.com/mittagessen/kraken) OCR engine in a docker container
- [Konfuzio](https://www.konfuzio.com) - Free Online OCR up to 2.000 pages per month and OCR API by [@atraining], see https://youtu.be/NZKUrKyFVA8 (code is not open)
- [ocr.space](https://ocr.space/) - Free Online OCR and OCR API by [@a9t9](https://github.com/A9T9) based on Tesseract (code is not open)
- [OCR4all](https://github.com/OCR4all/OCR4all) - Provides OCR services through web applications. Included Projects: [LAREX](https://github.com/chreul/LAREX), [OCRopus](https://github.com/tmbdev/ocropy), [calamari](https://github.com/ChWick/calamari) and [nashi](https://github.com/andbue/nashi).

### OCR evaluation

- [ISRI OCR Evaluation Tools](https://code.google.com/archive/p/isri-ocr-evaluation-tools/) with a [User Guide from 1996 :!:](https://github.com/eddieantonio/isri-ocr-evaluation-tools/blob/HEAD/user-guide.pdf)
  - [isri-ocr-evaluation-tools](https://github.com/eddieantonio/isri-ocr-evaluation-tools) - further development by [@eddieantonio](https://github.com/eddieantonio) (2015, 2016)
  - [ancientgreekocr-evaluation-tools](https://github.com/ryanfb/ancientgreekocr-ocr-evaluation-tools) - further development by [@nickjwhite](https://github.com/nickjwhite) (2013, 2014)
- [ocrevalUAtion](https://github.com/impactcentre/ocrevalUAtion) - Cross-format evaluation, CLI and GUI
- [ngram-ocr-eval](https://github.com/impactcentre/hackathon2014/tree/master/ngram-ocr-eval) - Brute and simple OCR evaluation using ngrams
- [quack](https://github.com/tokee/quack) - Quality-Assurance-tool for scans with corresponding ALTO-files

### OCR libraries by programming language

#### Crystal

- [tesseract-ocr](https://github.com/dannnylo/tesseract-ocr-crystal) - A Crystal wrapper for tesseract-ocr.

#### Elixir

- [tesseract_ocr](https://github.com/dannnylo/tesseract-ocr-elixir) - Elixir library wrapping the tesseract executable.

#### Go

- [gosseract](https://github.com/otiai10/gosseract) - Golang OCR library, wrapping Tesseract-ocr.

#### Java

- [Tess4J](https://github.com/nguyenq/tess4j) - Java Native Access bindings to Tesseract.
- [tess-two](https://github.com/rmtheis/tess-two) - Tools for compiling Tesseract on Android and Java API.

#### .Net

- [tesseract for .net](https://github.com/charlesw/tesseract) - A .Net wrapper for tesseract-ocr.

#### Object Pascal

- [TTesseractOCR4](https://github.com/r1me/TTesseractOCR4) - Object Pascal binding for tesseract-ocr 4.x.

#### PHP

- [Tesseract OCR for PHP](https://github.com/thiagoalessio/tesseract-ocr-for-php) - Tesseract PHP bindings.

#### Python

- [pytesseract](https://github.com/madmaze/pytesseract) - A Python wrapper for Google Tesseract.
- [pyocr](https://github.com/jflesch/pyocr) - A Python wrapper for Tesseract and Cuneiform.
- [ocrodjvu](https://github.com/jwilk/ocrodjvu) - A library and standalone tool for doing OCR on DjVu documents, wrapping Cuneiform, gocr, ocrad, ocropus and tesseract
- [tesserocr](https://github.com/sirfz/tesserocr) - A Python wrapper for the tesseract-ocr API

#### Javascript

- [ocracy](https://github.com/naptha/ocracy) - pure javascript lstm rnn implementation based on ocropus
- [gocr.js](https://github.com/antimatter15/gocr.js) - Javascript port (emscripten) of gocr
- [ocrad.js](https://github.com/antimatter15/ocrad.js) - Javascript port (emscripten) of ocrad
- [tesseract.js](https://github.com/naptha/tesseract.js) - Javascript port (emscripten) of Tesseract
- [node-tesseract-ocr](https://github.com/zapolnoch/node-tesseract-ocr) - A simple wrapper for the Tesseract OCR package.
- [node-tesseract-native](https://github.com/mdelete/node-tesseract-native) - C++ module for node providing OCR with tesseract and leptonica.

#### Ruby

- [rtesseract](https://github.com/dannnylo/rtesseract) - Ruby library wrapping the tesseract and imagemagick executables.
- [ruby-tesseract](https://github.com/meh/ruby-tesseract-ocr) - Native Tesseract bindings for Ruby MRI and JRuby
- [ocr_space](https://github.com/suyesh/ocr_space) - API wrapper for free ocr service ocr.space. Includes CLI

#### Rust

- [tesseract.rs](https://github.com/antimatter15/tesseract-rs) - Rust bindings for tesseract OCR.
- [leptess](https://crates.io/crates/leptess) - Productive and safe Rust bindings/wrappers for tesseract and leptonica.

#### R

- [tesseract](https://github.com/ropensci/tesseract) - R bindings for tesseract OCR.

#### Swift

- [Tesseract OCR iOS](https://github.com/gali8/Tesseract-OCR-iOS) - Swift and Objective-C wrapper for Tesseract OCR.
- [SwiftOCR](https://github.com/garnele007/SwiftOCR) - Fast and simple OCR library written in Swift. Optimized for recognizing short, one line long alphanumeric codes.

### OCR training tools

- [glyph-miner](https://github.com/benedikt-budig/glyph-miner) - A system for extracting glyphs from early typeset prints
- [ocrodeg](https://github.com/NVlabs/ocrodeg) - Document image degradation for OCR data augmentation

## Datasets

### Ground Truth

- [archiscribe-corpus](https://github.com/jbaiter/archiscribe-corpus) - >4,200 lines transcribed from 19th Century German prints via [archiscribe](https://archiscribe.jbaiter.de/) `CC-BY 4.0`
- [CIS OCR Test Set](https://github.com/cisocrgroup/Resources/tree/master/ocrtestset) - 2 example documents each in German/Latin/Greek with ground truth for [PoCoTo](https://github.com/cisocrgroup/PoCoTo)

* [Rescribe](https://github.com/rescribe/carolineminuscule-groundtruth) - Transcriptions of Caroline Minuscule Manuscripts `PDM 1.0`

- [CLTK](https://github.com/cltk) - Corpora from [Classical Language Toolkit](http://cltk.org/) `PDM 1.0`
- [DIVA-HisDB](https://diuf.unifr.ch/main/hisdoc/diva-hisdb) - 150 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> of three medieval manuscripts `CC-BY-NC 3.0`
- [EarlyPrintedBooks](https://github.com/chreul/OCR_Testdata_EarlyPrintedBooks) - ~8,800 lines from several early printed books `CC-BY-NC-SA 4.0`
- [EEBO-TCP](https://github.com/Anterotesis/historical-texts/tree/master/eebo-tcp) - 25,363 EEBO documents transcribed by [TCP](http://www.textcreationpartnership.org/tcp-eebo/) `PDM 1.0`
- [ECCO-TCP](https://github.com/Anterotesis/historical-texts/tree/master/ecco-tcp) - 2,188 ECCO documents transcribed by [TCP](http://www.textcreationpartnership.org/tcp-ecco/) `PDM 1.0`
- [eMOP-TCP](https://github.com/Early-Modern-OCR/TCP-ECCO-texts) - 2,188 ECCO-TCP documents, cleaned up by [eMOP](http://emop.tamu.edu/) `PDM 1.0`
- [Evans-TCP](https://github.com/Anterotesis/historical-texts/tree/master/evans-tcp) - 4,977 Evans documents transcribed by [TCP](http://www.textcreationpartnership.org/tcp-evans/)
- [FDHN](https://digi.kansalliskirjasto.fi/opendata/submit?set_language=en) - Finnish Digitised Historical Newspapers, [Paper](http://doi.org/10.1045/july2016-paakkonen), (free) [registration](https://digi.kansalliskirjasto.fi/opendata/submit?set_language=en) required, [Terms of Use](https://digi.kansalliskirjasto.fi/terms)
- [FROC-MSS](https://github.com/Jean-Baptiste-Camps/FROC-MSS) - 4 Old French Medieval Manuscripts `CC-BY 4.0`
- [GERMANA](https://www.prhlt.upv.es/wp/resource/the-germana-corpus) - 764 Spanish manuscript pages, (free) [registration](https://www.prhlt.upv.es/wp/resource/the-germana-corpus) required `non-commercial use only`
- [GT4HistOCR](https://doi.org/10.5281/zenodo.1344132) - Ground Truth for German Fraktur and Early Modern Latin `CC-BY 4.0`
- [imagessan](https://github.com/Shreeshrii/imagessan/) - Sanskrit images & ground truth (Devanagari script)
- [IMPACT-BHL](http://www.bhle.eu/en/results-of-the-collaboration-of-bhl-europe-and-impact) - 2,418 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from the Biodiversity Heritage Library, [XML@GitHub](https://github.com/impactcentre/groundtruth-bhl) `CC-BY 3.0`
- [IMPACT-BL](https://www.digitisation.eu/tools-resources/image-and-ground-truth-resources/impact-dataset-browser/?query=&search-filter-institution=BL&search-filter-language=&search-filter-script=&search-filter-year=) - 294 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from the British Library, (free) [registration](https://www.digitisation.eu/wp-login.php?action=register) required `PDM 1.0`
- [IMPACT-BNE](https://www.digitisation.eu/tools-resources/image-and-ground-truth-resources/impact-dataset-browser/?query=&search-filter-institution=BNE&search-filter-language=&search-filter-script=&search-filter-year=) - 215 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from the National Library of Spain, (free) [registration](https://www.digitisation.eu/wp-login.php?action=register) required, [XML@GitHub](https://github.com/impactcentre/groundtruth-spa) `CC-BY-NC-SA 4.0`
- [IMPACT-BNF](https://www.digitisation.eu/tools-resources/image-and-ground-truth-resources/impact-dataset-browser/?query=&search-filter-institution=BNE&search-filter-language=&search-filter-script=&search-filter-year=) - 151 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from the National Library of France, (free) [registration](https://www.digitisation.eu/wp-login.php?action=register) required `CC-BY-NC-SA 4.0`
- [IMPACT-KB](http://lab.kb.nl/dataset/ground-truth-impact-project#access) - 142 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from the National Library of the Netherlands `CC-BY 4.0`
- [IMPACT-NKC](https://www.digitisation.eu/tools-resources/image-and-ground-truth-resources/impact-dataset-browser/?query=&search-filter-institution=NKC&search-filter-language=&search-filter-script=&search-filter-year=) - 187 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from the Czech National Library, (free) [registration](https://www.digitisation.eu/wp-login.php?action=register) required `CC-BY-NC-SA 4.0`
- [IMPACT-NLB](https://www.digitisation.eu/tools-resources/image-and-ground-truth-resources/impact-dataset-browser/?query=&search-filter-institution=NLB&search-filter-language=&search-filter-script=&search-filter-year=) - 19 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from the National Library of Bulgaria, (free) [registration](https://www.digitisation.eu/wp-login.php?action=register) required `CC-BY-NC-ND 4.0`
- [IMPACT-NUK](https://www.digitisation.eu/tools-resources/image-and-ground-truth-resources/impact-dataset-browser/?query=&search-filter-institution=NUK&search-filter-language=&search-filter-script=&search-filter-year=) - 209 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from the National Library of Slovenia, (free) [registration](https://www.digitisation.eu/wp-login.php?action=register) required `CC-BY-NC-SA 4.0`
- [IMPACT-PSNC](http://dl.psnc.pl/activities/projekty/impact/results/) - 478 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> from four Polish digital libraries, [XML@GitHub](https://github.com/impactcentre/groundtruth-pol) `CC-BY 3.0`
- [LascivaRoma/lexical](https://github.com/lascivaroma/lexical) - Transcription of 19th century lexical resources for Latin learning
- [MJSynth](http://www.robots.ox.ac.uk/~vgg/data/text/) - 9m synthetic images covering 90k English words
- [OCR19thSAC](https://files.ifi.uzh.ch/cl/OCR19thSAC/) - 19,000 pages Swiss Alpine Club yearbooks transcribed via [Text+Berg digital](http://textberg.ch/site/en/welcome/) `CC-BY 4.0`
- [OCR-D](http://ocr-d.de/daten) - 180 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> of German historical prints from [OCR-D](http://ocr-d.de/) `CC-BY-SA 4.0`
- [OCR_GS_Data](https://github.com/OpenITI/OCR_GS_Data) - Double-checked Arabic Gold Standard from [OpenITI](https://github.com/OpenITI)
- [old-books](https://github.com/PedroBarcha/old-books-dataset) - 322 old books from [Project Gutenberg](https://www.gutenberg.org/) `GPL 3.0`
- [PRImA-ENP](http://www.primaresearch.org/datasets/ENP) - 528 pages<sup>[PAGE-XML](https://github.com/PRImA-Research-Lab/PAGE-XML)</sup> historic newspapers from [Europeana Newspapers](http://www.europeana-newspapers.eu/), (free) [registration](http://www.primaresearch.org/register) required `PDM 1.0`
- [RODRIGO](https://www.prhlt.upv.es/wp/resource/the-rodrigo-corpus) - 853 Spanish manuscript pages, (free) [registration](https://www.prhlt.upv.es/wp/resource/the-rodrigo-corpus) required `non-commercial use only`
- [Toebler-OCR](https://github.com/PonteIneptique/toebler-ocr) - (Kraken) Ground Truth transcription of few pages of the Tobler-Lommatzsch: Altfranzösisches Wörterbuch

## Video Text Spotting

- [VideoTextSCM](https://github.com/lsabrinax/VideoTextSCM)
- [TransDETR](https://github.com/weijiawu/TransDETR)
- [YORO](https://github.com/hikopensource/DAVAR-Lab-OCR/tree/main/demo/videotext/yoro) ([paper:2021](https://arxiv.org/pdf/1903.03299.pdf))

## Font detection

- [typefont](https://github.com/Vasile-Peste/Typefont) - The first open-source library that detects the font of a text in a image.

## Optical Character Recognition Engines and Frameworks

- [DAVAR-lab-OCR](https://github.com/hikopensource/davar-lab-ocr)
- [CRNN.tf2](https://github.com/FLming/CRNN.tf2)
- [ocr.pytorch](https://github.com/courao/ocr.pytorch)
- [PytorchOCR](https://github.com/WenmuZhou/PytorchOCR)
- [MMOCR](https://github.com/open-mmlab/mmocr)
- [doctr](https://github.com/mindee/doctr)
- [Master OCR](https://github.com/jiangxiluning/MASTER-TF)
- [xiaofengShi/CHINESE-OCR](https://github.com/xiaofengShi/CHINESE-OCR)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Urdu-Ocr](https://github.com/HassamChundrigar/Urdu-Ocr)
- [ocr.pytorch](https://github.com/courao/ocr.pytorch)
- [ocular](https://github.com/ndnlp/ocular) - Ocular is a state-of-the-art historical OCR system.
- [OCR++](https://github.com/mayank4490/OCR-plus-plus)
- [pytextrator](https://github.com/danwald/pytextractor) - python ocr using tesseract/ with EAST opencv detector
- [OCR-D](https://ocr-d.github.io/)
- [ocrd_tesserocr](https://github.com/OCR-D/ocrd_tesserocr)
- [Deeplearning-OCR](https://github.com/vinayakkailas/Deeplearning-OCR)
- [PICCL](https://github.com/LanguageMachines/PICCL)
- [cnn_lstm_ctc_ocr](https://github.com/weinman/cnn_lstm_ctc_ocr) - Tensorflow-based CNN+LSTM trained with CTC-loss for OCR.
- [PassportScanner](https://github.com/evermeer/PassportScanner) - Scan the MRZ code of a passport and extract the firstname, lastname, passport number, nationality, date of birth, expiration date and personal numer.
- [pannous/tensorflow-ocr](https://github.com/pannous/tensorflow-ocr) - OCR using tensorflow with attention.
- [BowieHsu/tensorflow_ocr](https://github.com/BowieHsu/tensorflow_ocr) - OCR detection implement with tensorflow v1.4.
- [GRCNN-for-OCR](https://github.com/Jianfeng1991/GRCNN-for-OCR) - This is the implementation of the paper "Gated Recurrent Convolution Neural Network for OCR"
- [go-ocr](https://github.com/maxim2266/go-ocr) - A tool for extracting text from scanned documents (via OCR), with user-defined post-processing.
- [insightocr](https://github.com/deepinsight/insightocr) - MXNet OCR implementation. Including text recognition and detection.
- [ocr_densenet](https://github.com/yinchangchang/ocr_densenet) - The first Xi'an Jiaotong University Artificial Intelligence Practice Contest (2018AI Practice Contest - Picture Text Recognition) first; only use the densenet to identify the Chinese characters
- [CNN_LSTM_CTC_Tensorflow](https://github.com/watsonyanghx/CNN_LSTM_CTC_Tensorflow) - CNN+LSTM+CTC based OCR implemented using tensorflow.
- [tmbdev/clstm](https://github.com/tmbdev/clstm) - A small C++ implementation of LSTM networks, focused on OCR.
- [VistaOCR](https://github.com/isi-vista/VistaOCR)
- [tesseract.js](https://github.com/naptha/tesseract.js)
- [Tesseract](https://github.com/tesseract-ocr/tesseract)
- [kaldi](https://github.com/kaldi-asr/kaldi)
- [ocropus3](https://github.com/NVlabs/ocropus3) - Repository collecting all the submodules for the new PyTorch-based OCR System.
- [calamari](https://github.com/Calamari-OCR/calamari)
- [ocropy](https://github.com/tmbdev/ocropy) - Python-based tools for document analysis and OCR
- [chinese_ocr](https://github.com/YCG09/chinese_ocr)
- [deep_ocr](https://github.com/JinpengLI/deep_ocr) - make a better chinese character recognition OCR than tesseract.
- [ocular](https://github.com/tberg12/ocular)
- [textDetectionWithScriptID](https://github.com/isi-vista/textDetectionWithScriptID)
- [transcribus](https://transkribus.eu/Transkribus/)
- [FastText](https://fasttext.cc/) - Library for efficient text classification and representation learning
- [GOCR](http://www-e.uni-magdeburg.de/jschulen/ocr/)
- [Ocrad](https://www.gnu.org/software/ocrad/)
- [franc](https://github.com/wooorm/franc) - Natural language detection
- [ocrfeeder](https://github.com/GNOME/ocrfeeder)
- [emedvedev/attention-ocr](https://github.com/emedvedev/attention-ocr) - A Tensorflow model for text recognition (CNN + seq2seq with visual attention) available as a Python package and compatible with Google Cloud ML Engine.
- [da03/attention-ocr](https://github.com/da03/Attention-OCR) - Visual Attention based OCR
- [dhlab-epfl/dhSegment](https://github.com/dhlab-epfl/dhSegment) - Generic framework for historical document processing
- https://github.com/mawanda-jun/TableTrainNet
- https://github.com/kermitt2/delft
- https://github.com/chulwoopack/docstrum
- [grobid](https://github.com/kermitt2/grobid) - A machine learning software for extracting information from scholarly documents
- [lapdftext](http://bmkeg.github.io/lapdftext/) - LA-PDFText is a system for extracting accurate text from PDF-based research articles
- https://github.com/beratkurar/textline-segmentation-using-fcn
- https://github.com/OCR4all
- https://github.com/OCR4all/LAREX
- https://github.com/OCR4all/OCR4all
- https://github.com/andbue/nashi
- http://kraken.re/
- [kraken](https://github.com/mittagessen/kraken)
- [gosseract](https://github.com/otiai10/gosseract) - Go package for OCR (Optical Character Recognition), by using Tesseract C++ library.
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Ready-to-use OCR with 40+ languages supported including Chinese, Japanese, Korean and Thai.
- [invoice-scanner-react-native](https://github.com/burhanuday/invoice-scanner-react-native)
- [Arabic-OCR](https://github.com/HusseinYoussef/Arabic-OCR)

## Awesome lists

- https://github.com/whitelok/image-text-localization-recognition
- [Awesome-Scene-Text-Recognition](https://github.com/chongyangtao/Awesome-Scene-Text-Recognition) -
  A curated list of resources dedicated to scene text localization and recognition
- [awesome-deep-text-detection-recognition](https://github.com/hwalsuklee/awesome-deep-text-detection-recognition)
- https://github.com/kurapan/awesome-scene-text
- [kba/awesome-ocr](https://github.com/kba/awesome-ocr)
- [perfectspr/awesome-ocr](https://github.com/perfectspr/awesome-ocr)
- https://github.com/ZumingHuang/awesome-ocr-resources
- https://github.com/chongyangtao/Awesome-Scene-Text-Recognition
- https://github.com/whitelok/image-text-localization-recognition
- https://github.com/hwalsuklee/awesome-deep-text-detection-recognition
- https://github.com/wanghaisheng/awesome-ocr
- https://github.com/Jyouhou/SceneTextPapers
- https://github.com/jyhengcoder/myOCR
- https://github.com/hwalsuklee/awesome-deep-text-detection-recognition
- https://github.com/tangzhenyu/Scene-Text-Understanding
- https://github.com/whitelok/image-text-localization-recognition
- https://github.com/kba/awesome-ocr
- https://github.com/soumendra/awesome-ocr
- [chongyangtao/Awesome-Scene-Text-Recognition](https://github.com/chongyangtao/Awesome-Scene-Text-Recognition) - Papers and datasets

## Proprietary OCR Engines

- [ABBYY](https://www.abbyy.com/en-us/)
- [Omnipage](https://www.nuance.com/print-capture-and-pdf-solutions.html)
- [Clova.ai](https://demo.ocr.clova.ai/)
- [Konfuzio](https://konfuzio.com/en/)

## Cloud based OCR Engines (SaaS)

- [thehive.ai](https://thehive.ai/hive-ocr-solutions)
- [impira](https://www.impira.com/try/smarter-ocr)
- [AWS Textracet](https://aws.amazon.com/textract/)
- [Nanonets](https://nanonets.com/ocr-api/)
- [docparser](https://docparser.com/
- [ocrolus](https://www.ocrolus.com/)
- [Butler Labs](https://www.butlerlabs.ai/)

## File formats and tools

- [nw-page-editor](https://github.com/mauvilsa/nw-page-editor) - Simple app for visual editing of Page XML files
- [hocr](http://kba.cloud/hocr-spec/1.2/)
- [alto](https://github.com/altoxml)
- [PageXML](https://github.com/PRImA-Research-Lab/PAGE-XML)
- [ocr-fileformat](https://github.com/UB-Mannheim/ocr-fileformat) - Validate and transform various OCR file formats
- [hocr-tools](https://github.com/tmbdev/hocr-tools) - Tools for manipulating and evaluating the hOCR format for representing multi-lingual OCR results by embedding them into HTML.

## Datasets

- http://www.iapr-tc11.org/mediawiki/index.php/Datasets_List
- https://icdar2019.org/competitions-2/
- https://rrc.cvc.uab.es/#
- https://lionbridge.ai/datasets/15-best-ocr-handwriting-datasets/
- https://github.com/xylcbd/ocr-open-dataset
- ICDAR datasets
- https://github.com/OpenArabic/OCR_GS_Data
- https://github.com/cs-chan/Total-Text-Dataset
- [scenetext](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) - This is a synthetically generated dataset, in which word instances are placed in natural scene images, while taking into account the scene layout.
- [Total-Text-Dataset](https://github.com/cs-chan/Total-Text-Dataset)
- [ocr-open-dataset](https://github.com/xylcbd/ocr-open-dataset)

## Data augmentation and Synthetic data generation

- [DocCreator](http://doc-creator.labri.fr/) - DIAR software for synthetic document image and groundtruth generation, with various degradation models for data augmentation.
- [Scene-Text-Image-Transformer](https://github.com/Canjie-Luo/Scene-Text-Image-Transformer) - Scene Text Image Transformer
- [Belval/TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) - A synthetic data generator for text recognition
- [Sanster/text_renderer](https://github.com/Sanster/text_renderer)
- [awesome-SynthText](https://github.com/TianzhongSong/awesome-SynthText)
- [Text-Image-Augmentation](https://github.com/Canjie-Luo/Text-Image-Augmentation)
- [UnrealText](https://github.com/Jyouhou/UnrealText)
- [SynthText_Chinese_version](https://github.com/JarveeLee/SynthText_Chinese_version)

## Pre OCR Processing

- [ajgalleo/document-image-binarization](https://github.com/ajgallego/document-image-binarization)
- [PRLib](https://github.com/leha-bot/PRLib) - Pre-Recognize Library - library with algorithms for improving OCR quality.
- [sbb_binarization](https://github.com/qurator-spk/sbb_binarization) -

## Post OCR Correction

- [KBNLresearch/ochre](https://github.com/KBNLresearch/ochre) - Toolbox for OCR post-correction
- [cisocrgroup/PoCoTo](https://github.com/cisocrgroup/PoCoTo) - The CIS OCR PostCorrectionTool
- [afterscan](http://www.afterscan.com/)

## Benchmarks

- [TedEval](https://github.com/clovaai/TedEval)
- [clovaai/deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark) - Text recognition (optical character recognition) with deep learning methods.
- [dinglehopper](https://github.com/qurator-spk/dinglehopper) - dinglehopper is an OCR evaluation tool and reads ALTO, PAGE and text files.
- [CLEval](https://github.com/clovaai/CLEval)

## misc

- [ocrodeg](https://github.com/NVlabs/ocrodeg) - a small Python library implementing document image degradation for data augmentation for handwriting recognition and OCR applications.
- [scantailor](https://github.com/scantailor/scantailor) - Scan Tailor is an interactive post-processing tool for scanned pages.
- [jlsutherland/doc2text](https://github.com/jlsutherland/doc2text) - help researchers fix these errors and extract the highest quality text from their pdfs as possible.
- [mauvilsa/nw-page-editor](https://github.com/mauvilsa/nw-page-editor) - Simple app for visual editing of Page XML files.
- [Transkribus](https://transkribus.eu/Transkribus/) - Transkribus is a comprehensive platform for the digitisation, AI-powered recognition, transcription and searching of historical documents.
- http://projectnaptha.com/
- https://github.com/4lex4/scantailor-advanced
- [open-semantic-search](https://github.com/opensemanticsearch/open-semantic-search) - Open Semantic Search Engine and Open Source Text Mining & Text Analytics platform (Integrates ETL for document processing, OCR for images & PDF, named entity recognition for persons, organizations & locations, metadata management by thesaurus & ontologies, search user interface & search apps for fulltext search, faceted search & knowledge graph)
- [ocrserver](https://github.com/otiai10/ocrserver) - A simple OCR API server, seriously easy to be deployed by Docker, on Heroku as well
- [cosc428-structor](https://github.com/chadoliver/cosc428-structor) - ~1000 book pages + OpenCV + python = page regions identified as paragraphs, lines, images, captions, etc.
- [nidaba](https://github.com/openphilology/nidaba/) - An expandable and scalable OCR pipeline
- https://github.com/MaybeShewill-CV/CRNN_Tensorflow
- [OCRmyPDF](https://github.com/jbarlow83/OCRmyPDF)

## Literature

### OCR-related publication and link lists

- [IMPACT: Tools for text digitisation](http://www.digitisation.eu/tools-resources/tools-for-text-digitisation/) - List of tools software projects related, some related to OCR
- [OCR-D](https://www.zotero.org/groups/ocr-d) - List of OCR-related academic articles in the context of the [OCR-D](http://www.ocr-d.de/) project. :de:
- [Mendeley Group "OCR - Optical Character Recognition"](https://www.mendeley.com/groups/752871/ocr-optical-character-recognition/) - Collection of 34 papers on OCR
- [eadh.org projects](http://eadh.org/projects) - List of Digital Humanities-related projects in Europe, some related to OCR
- [Wikipedia: Comparison of optical character recognition software](https://en.wikipedia.org/wiki/Comparison_of_optical_character_recognition_software)
- [OCR [and Deep Learning]](http://handong1587.github.io/deep_learning/2015/10/09/ocr.html) by [@handong1587](https://github.com/handong1587/)
- [Ocropus Wiki: Publications](https://github.com/tmbdev/ocropy/wiki/Publications)

### Blog Posts and Tutorials

- [Tesseract Blends Old and New OCR Technology](https://github.com/tesseract-ocr/docs/tree/master/das_tutorial2016) (2016) [@theraysmith](https://github.com/theraysmith)
  - Tutorial@DAS2016, Updated "What You Always Wanted to Know" slides
- [What You Always Wanted To Know About Tesseract](https://drive.google.com/folderview?id=0B7l10Bj_LprhQnpSRkpGMGV2eE0&usp#list) (2014) [@theraysmith](https://github.com/theraysmith)
  - Tutorial@DAS2014, includes demos
- [Extracting text from an image using Ocropus](http://www.danvk.org/2015/01/09/extracting-text-from-an-image-using-ocropus.html) (2015)
- [Training an Ocropus OCR model](http://www.danvk.org/2015/01/11/training-an-ocropus-ocr-model.html) (2015) [@danvk](https://github.com/danvk)
- [Ocropus Wiki: Compute errors and confusions](https://github.com/tmbdev/ocropy/wiki/Compute-errors-and-confusions) (2016) [@zuphilip](https://github.com/zuphilip)
- [Ocropus Wiki: Working with Ground Truth](https://github.com/tmbdev/ocropy/wiki/Compute-errors-and-confusion://github.com/tmbdev/ocropy/wiki/Working-with-Ground-Truth) (2016) [@zuphilip](https://github.com/zuphilip)
- [OCRopus](https://comsys.informatik.uni-kiel.de/lang/de/res/ocropus/) (2016) [@jze](https://github.com/jze)
  - mostly on column separation in ocropus
- [10 Tips for making your OCR project succeed](http://blog.kbresearch.nl/2013/12/12/10-tips-for-making-your-ocr-project-succeed/) (2013) [@cneud](https://github.com/cneud)
  - general things to consider for OCR projects
- [Overview of LEADTOOLS Image Cleanup and Pre-processing SDK Technology](https://www.leadtools.com/sdk/image-processing/document) -
  - feature list for a commercial image pre-processing library; has nice before-after samples for pre-processing steps related to OCR
- [Extracting Text from PDFs; Doing OCR; all within R](https://electricarchaeology.ca/2014/07/15/doing-ocr-within-r/) [@shawngraham](https://github.com/shawngraham)
  - How to work with OCR from PDFs in the [R programming environment](https://www.r-project.org/)
- [Tutorial: Command-line OCR on a Mac](http://benschmidt.org/dighist13/?page_id=129) [@bmschmidt](https://github.com/bmschmidt)
  - Tutorial on how to run tesseract in Mac OSX
- [Practical Expercience with OCRopus Model Training](https://comsys.informatik.uni-kiel.de/lang/de/res/practical-expercience-with-ocropus-model-training/) (2016) [@jze](https://github.com/jze)
- [Homemade Manuscript OCR (1): OCRopy](http://graal.hypotheses.org/786) (2017) [@Jean-Baptiste-Camps](https://github.com/Jean-Baptiste-Camps)
  - Tutorial on applying OCR to medieval manuscripts with OCRopy
- [Optimizing Binarization for OCRopus](https://comsys.informatik.uni-kiel.de/lang/de/res/optimizing-binarization-for-ocropus/) (2017) [@jze](https://github.com/jze)
- [Prototype demo for OCR postfix in Danish Newspapers](https://sbdevel.wordpress.com/2016/11/15/prototype-demo-for-ocr-postfix-in-danish-newspapers/) (2016) [@thomasegense](https://github.com/thomasegense)
- [How Can I OCR My Dictionary?](https://digilex.hypotheses.org/153) (2016) [@JessedeDoes](https://github.com/JessedeDoes)
- ["Needlessly complex" blog](https://mzucker.github.io/) (2016) [@mzucker](https://github.com/mzucker). Several image processing how-tos (Python based), particularly:
  - [Page dewarping](https://mzucker.github.io/2016/08/15/page-dewarping.html) ([code](https://github.com/mzucker/page_dewarp))
  - [Compressing and enhancing hand-written notes](https://mzucker.github.io/2016/09/20/noteshrink.html) ([code](https://github.com/mzucker/noteshrink))
  - [Unprojecting text with ellipses](https://mzucker.github.io/2016/10/11/unprojecting-text-with-ellipses.html) ([code](https://github.com/mzucker/unproject_text))
- [(Open-Source-)OCR-Workflows](https://edoc.bbaw.de/frontdoor/index/index/docId/2786) (2017) [@wrznr](https://github.com/wrznr) :de: overview of the state of the art in open source OCR and related technologies (binarisation, deskewing, layout recognition, etc.), lots of example images and information on the [@OCR-D](https://github.com/OCR-D) project.
- [A gentle introduction to OCR](https://towardsdatascience.com/a-gentle-introduction-to-ocr-ee1469a201aa) (2018) [@shgidi](https://github.com/shgidi)
- [Worauf kann ich mich verlassen? Arbeiten mit digitalisierten Quellen, Teil 1: OCR](https://blog.ub.unibas.ch/2019/06/04/worauf-kann-ich-mich-verlassen-arbeiten-mit-digitalisierten-quellen-teil-1-ocr/) (2019) [@eliaskreyenbuehl](https://github.com/eliaskreyenbuehl) :de: A reflection/criticism on OCR quality, OCR pitfalls in Fraktur fonts.

### OCR Showcases

- [abbyy-finereader-ocr-senate](https://github.com/dannguyen/abbyy-finereader-ocr-senate) - Using OCR to parse scanned Senate Financial Disclosure forms.
- [cvOCR](https://github.com/Halfish/cvOCR) - An OCR system for recognizing resume or cv text, implemented in Python and C and based on tesseract
- [MathOCR](https://github.com/chungkwong/MathOCR) - A printed scientific document recognition system, **pre-alpha**

### Academic articles

#### 2011 and before

- [High performance document layout analysis](http://www.dfki.de/web/research/publications/renameFileForDownload?filename=HighPerfDocLayoutAna.pdf&file_id=uploads_552) (2003) Breuel
- [Adaptive degraded document image binarization](http://doai.io/10.1016/j.patcog.2005.09.010) (2006) Gatos, Pratikakis, Perantonis
- [[Internship Report]](http://www.madm.eu/_media/theses/ocropusgupta.pdf) (2007) Gupta
- [OCRopus Addons (Internship Report)](http://madm.dfki.de/_media/theses/ocropusdantrey.pdf) (2007) Dantrey

#### 2012

- [Local Logistic Classifiers for Large Scale Learning](http://www.academia.edu/2959462/Local_Logistic_Classifiers_for_Large_Scale_Learning) (2012) Yousefi, Breuel

#### 2013

- [High Performance OCR for Printed English and Fraktur using LSTM Networks](http://staffhome.ecm.uwa.edu.au/~00082689/papers/Breuel-LSTM-OCR-ICDAR13.pdf) (2013) Breuel, Ul-Hasan, Mayce Al Azawi. Shafait
- [Can we build language-independent OCR using LSTM networks?](https://www.researchgate.net/publication/260341307_Can_we_build_language-independent_OCR_using_LSTM_networks) (2013) Ul-Hasan, Breuel
- [Offline Printed Urdu Nastaleeq Script Recognition with Bidirectional LSTM Networks](http://staffhome.ecm.uwa.edu.au/~00082689/papers/Adnan-Urdu-OCR-ICDAR13.pdf) (2013) Ul-Hasan, Ahmed, Rashid, Shafait, Breuel

#### 2014

- [OCR of historical printings of Latin texts: Problems, Prospects, Progress.](http://www.springmann.net/papers/2014-04-07-DATeCH2014-springmann.pdf) (2014) Springmann, Najock, Morgenroth, Schmid, Gotscharek, Fink
- [Correcting Noisy OCR: Context beats Confusion](http://dx.doi.org/10.1145/2595188.2595200) (2014) Evershed, Fitch

#### 2015

- [TypeWright: An Experiment in Participatory Curation](http://www.digitalhumanities.org/dhq/vol/9/4/000220/000220.html) (2015) Bilansky
  - On crowd-sourcing OCR postcorrection
- [Benchmarking of LSTM Networks](http://arxiv.org/abs/1508.02774) (2015) Breuel
- [Recognition of Historical Greek Polytonic Scripts Using LSTM](http://users.iit.demokritos.gr/~bgat/OldDocPro/05_paper_305.pdf) (2015) Simistira, Ul-Hassan, Papavassiliou, Basilis Gatos, Katsouros, Liwicki
- [A Segmentation-Free Approach for Printed Devanagari Script Recognition](https://www.researchgate.net/publication/280777081_A_Segmentation-Free_Approach_for_Printed_Devanagari_Script_Recognition) (2015) Karayil, Ul-Hasan, Breuel
- [A Sequence Learning Approach for Multiple Script Identification](https://www.researchgate.net/publication/280777013_A_Sequence_Learning_Approach_for_Multiple_Script_Identification) (2015) Ul-Hasan, Afzal, Shfait, Liwicki, Breuel

#### 2016

- [Important New Developments in Arabographic Optical Character Recognition (OCR)](https://arxiv.org/abs/1703.09550) (2016) Romanov, Miller, Savant, Kiessling
  - on [kraken](#ocr-engines)
  - using [OpenArabic/OCR_GS_Data](https://github.com/OpenArabic/OCR_GS_Data) for ground truth data
- [OCR of historical printings with an application to building diachronic corpora: A case study using the RIDGES herbal corpus](https://arxiv.org/abs/1608.02153) (2016) Springmann, Lüdeling
- [Automatic quality evaluation and (semi-) automatic improvement of mixed models for OCR on historical documents](http://arxiv.org/abs/1606.05157) (2016) Springmann, Fink, Schulz
- [Generic Text Recognition using Long Short-Term Memory Networks](https://kluedo.ub.uni-kl.de/frontdoor/index/index/docId/4353) (2016) Ul-Hasan -- Ph.D Thesis
- [OCRoRACT: A Sequence Learning OCR System Trained on Isolated Characters](https://www.researchgate.net/publication/294575734_OCRoRACT_A_Sequence_Learning_OCR_System_Trained_on_Isolated_Characters) (2016) Dengel, Ul-Hasan, Bukhari
- [Recursive Recurrent Nets with Attention Modeling for OCR in the Wild](https://arxiv.org/abs/1603.03101) (2016) Lee, Osindero
- [paper:2016](https://arxiv.org/pdf/1609.03605.pdf)
- [text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)
- [yizt/keras-ctpn](https://github.com/yizt/keras-ctpn)
- [tianzhi0549/CTPN](https://github.com/tianzhi0549/CTPN) - Detecting Text in Natural Image with Connectionist Text Proposal Network
- [paper:2016](https://arxiv.org/abs/1611.06779)
- [TextBoxes (official)](https://github.com/MhLiao/TextBoxes)
- [TextBoxes-TensorFlow](https://github.com/gxd1994/TextBoxes-TensorFlow) - TextBoxes re-implement using tensorflow
- [zj463261929/TextBoxes](https://github.com/zj463261929/TextBoxes) - TextBoxes: A Fast Text Detector with a Single Deep Neural Network
- [shinjayne/textboxes](https://github.com/shinjayne/textboxes) - Textboxes implementation with Tensorflow (python)
- [shinTB](https://github.com/shinjayne/shinTB) - Textboxes : Image Text Detection Model : python package (tensorflow)

#### 2017

- [Telugu OCR Framework using Deep Learning](https://arxiv.org/abs/1509.05962) (2015/2017) [Achanta](https://github.com/rakeshvar), Hastie
  - see also [TeluguOCR](https://github.com/TeluguOCR), [banti_telugu_ocr](https://github.com/TeluguOCR/banti_telugu_ocr), [chamanti_ocr](https://github.com/rakeshvar/chamanti_ocr), [#49](https://github.com/kba/awesome-ocr/issues/49)
- [EAST](https://github.com/argman/EAST)(official) - (tf1/py2) A tensorflow implementation of EAST text detector
- [AdvancedEAST](https://github.com/huoyijie/AdvancedEAST) - (tf1/py2) AdvancedEAST is an algorithm used for Scene image text detect, which is primarily based on EAST, and the significant improvement was also made, which make long text predictions more accurate.
- [kurapan/EAST](https://github.com/kurapan/EAST) Implementation of EAST scene text detector in Keras
- [songdejia/EAST](https://github.com/songdejia/EAST) - This is a pytorch re-implementation of EAST: An Efficient and Accurate Scene Text Detector.
- [HaozhengLi/EAST_ICPR](https://github.com/HaozhengLi/EAST_ICPR) - Forked from argman/EAST for the ICPR MTWI 2018 CHALLENGE
- [deepthinking-qichao/EAST_ICPR2018](https://github.com/deepthinking-qichao/EAST_ICPR2018)
- [SakuraRiven/EAST](https://github.com/SakuraRiven/EAST)
- [EAST-Detector-for-text-detection-using-OpenCV](https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV) - Text Detection from images using OpenCV
- [easy-EAST](https://github.com/che220/easy-EAST)

#### 2018

- [A Two-Stage Method for Text Line Detection in Historical Documents](https://arxiv.org/abs/1802.03345) (2018) [Grüning](https://github.com/TobiasGruening), Leifert, Strauß, Labahn. Code available at https://github.com/TobiasGruening/ARU-Net
- [tensorflow_PSENet](https://github.com/liuheng92/tensorflow_PSENet) - This is a tensorflow re-implementation of PSENet: Shape Robust Text Detection with Progressive Scale Expansion Network
- [PAN-PSEnet](https://github.com/rahzaazhar/PAN-PSEnet)
- [PSENet](https://github.com/whai362/PSENet) - Shape Robust Text Detection with Progressive Scale Expansion Network.
- FOTS [paper:2018](https://arxiv.org/pdf/1801.01671.pdf)
- [FOTS](https://github.com/xieyufei1993/FOTS) - An Implementation of the FOTS: Fast Oriented Text Spotting with a Unified Network.
- [FOTS_OCR](https://github.com/Masao-Taketani/FOTS_OCR)
- TextBoxes++ [paper:2018](https://arxiv.org/abs/1801.02765)
- [TextBoxes_plusplus (offical)](https://github.com/MhLiao/TextBoxes_plusplus) TextBoxes++: A Single-Shot Oriented Scene Text Detector
- [Shun14/TextBoxes_plusplus_Tensorflo](https://github.com/Shun14/TextBoxes_plusplus_Tensorflow) - Textboxes_plusplus implementation with Tensorflow (python)

#### 2019

- RAFT [paper:2019](https://arxiv.org/pdf/1904.01941.pdf)
- [CRAFT-pytorch (official)](https://github.com/clovaai/CRAFT-pytorch) - Pytorch implementation of CRAFT text detector.
- [autonise/CRAFT-Remade](https://github.com/autonise/CRAFT-Remade)
- [s3nh/pytorch-text-recognition](https://github.com/s3nh/pytorch-text-recognition)
- [backtime92/CRAFT-Reimplementation](https://github.com/backtime92/CRAFT-Reimplementation)
- [fcakyon/craft-text-detector](https://github.com/fcakyon/craft-text-detector) - PyTorch implementation of CRAFT
- [YongWookHa/craft-text-detector](https://github.com/YongWookHa/craft-text-detector)
- [faustomorales/keras-ocr](https://github.com/faustomorales/keras-ocr) - A packaged and flexible version of the CRAFT text detector and Keras CRNN recognition model.
- [fcakyon/craft-text-detector](https://github.com/fcakyon/craft-text-detector)

#### 2020

- ABCNet [paper:2020](https://arxiv.org/pdf/2002.10200.pdf)
- [AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
- https://github.com/Yuliang-Liu/bezier_curve_text_spotting
- https://github.com/quangvy2703/ABCNet-ESRGAN-SRTEXT
- https://github.com/Pxtri2156/AdelaiDet_v2
- https://github.com/zhubinQAQ/Ins
