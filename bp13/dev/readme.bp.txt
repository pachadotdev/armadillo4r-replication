Christiane Baumeister and Gert Peersman, "The Role of Time-Varying Price 
Elasticities in Accounting for Volatility Changes in the Crude Oil 
Market", Journal of Applied Econometrics, Vol. 28, No. 7, 2013, pp. 
1087-1109.

This directory contains the following 3 files that are described in
detail below:

  bp-appendix.pdf
  bp-data.zip
  bp-programs.zip

*************************************************************************
Note: The datasets contained in bp-data.zip are not to be used with the 
      MATLAB codes contained in bp-programs.zip. The data files required 
      for running the codes are included in the bp-programs.zip file. 
      Further details can be found in the readme file in bp-programs.zip. 
*************************************************************************

1. APPENDIX (bp-appendix.pdf)

The file bp-appendix.pdf contains a detailed description of the data
sources, the estimation procedures for the econometric model, the
results from a Monte Carlo study, evidence for time variation,
sensitivity analyses and additional results.

This PDF file should be readable using Acrobat Reader or any other
program for reading PDF files.


2. DATA (bp-data.zip)

The file bp-data.zip contains files with the data used in the paper.
The data are stored as ASCII files in DOS format. Unix/Linux users
should use "unzip -a". The variables are stored as columns.

bp_bench.txt:   Contains quarterly data on world oil production, the real
                price of crude oil, and world industrial production from 
                1947Q1 to 2010Q4 (256 observations).
                Source: EIA, Oil & Gas Journal, United Nations Monthly 
                Bulletin of Statistics and authors' calculations.

bp_kilian.txt:  Contains quarterly data on world oil production, the real
                price of crude oil, and a global business cycle index from 
                1967Q4 to 2010Q4 (173 observations).
                Source: EIA, Oil & Gas Journal and Lutz Kilian's website 
                (http://www-personal.umich.edu/~lkilian/reaupdate.txt).

bp_volume.txt:  Contains monthly data on the total trading volume of WTI
                futures contracts with less than one year to maturity from
                1983M4 to 2010M12 (333 observations). 
                Source: PriceData.com and authors' calculations.

bp_rig.txt:     Contains monthly data on worldwide oil rig counts from 
                1975M1 to 2010M12 (432 observations). 
                Source: Baker and Hughes Inc.

bp_caputil.txt: Contains annual data on global capacity utilization rates
                in crude oil production from 1974 to 2010 (37 observations). 
                Source: IMF Economic Outlook and authors' calculations.


3. PROGRAMS (bp-programs.zip)

The file bp-programs.zip contains the MATLAB programs used in the
estimation of the time-varying parameter VAR model. We used version
Matlab R2010a to run the programs. The codes are prepared to be run in
the directory C:\BP2012codes.  Thus, the folder 'BP2012codes' needs to
be copied on the C-drive.

The script file contained in the folder 'BP2012codes' is called
oilVAR3world_setup.m.  This file calls all the other functions. A
detailed desription of the m-files is provided in the readme file in
the folder 'BP2012codes' in bp-programs.zip.

Some of the files in bp-programs.zip are binary files, so Unix/Linux
users should *not* use the "-a" option of unzip.

Please address questions to:

Christiane Baumeister
Bank of Canada
International Economic Analysis Department
234 Wellington Street
Ottawa ON K1A 0G9 
Canada

cbaumeister <AT> bankofcanada.ca
