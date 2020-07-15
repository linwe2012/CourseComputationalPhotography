:: A script to run all tests
:: make sure there is results folder
"./BoxFilter" ./OpenCVHW1/Lena.png ./results/BoxFilter.png 15 4
"./GaussianFilter" ./OpenCVHW1/Lena.png ./results/GaussianFilter.png 15
"./BilateralFilter" ./OpenCVHW1/Lena.png ./results/BilateralFilter.png 9 20
"./MedianFilterNaive" ./OpenCVHW1/Lena.png ./results/MedianFilterNaive.png 7 7
"./MedianFilter" ./OpenCVHW1/Lena.png ./results/MedianFilter.png 5 5
"./MedianFilterMultiThread" ./OpenCVHW1/Lena_sp.png ./results/MedianFilterMultiThread.png 59 59