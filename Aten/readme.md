## geting a benchmark on matrix multiplication 

"Numpy is almost a 100 times improvement from strassen, it does run on a single thread but I dont know how to make it parallel :( "

- Use divide and conquer --> its just strassen and it SUCKS!! [X]
- Coppersmith-Winogard
- Blas method 
- using vector intrinsics ! (it may improve the performance __m256 reg)
