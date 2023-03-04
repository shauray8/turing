## geting a benchmark on matrix multiplication 

### Matrix Multiplication is the most important part of a neural network !!
"Numpy is almost a 100 times improvement from strassen, it does run on a single thread but I dont know how to make it parallel :( "


- Use divide and conquer --> its just strassen and it SUCKS!! [X]
- Coppersmith-Winogard
- Blas method 
- using vector intrinsics ! (it may improve the performance __ __m256 reg __)
  - its TOUGH I'll try implimenting this for a couple of days if I fail to do so I will shift to other 
  more sophesticated algorithms !'
- I give up for now not doing it for a couple of days focusing on the tensor part of it !



start.cu -> use cuda to manufacture a solution for matrix multiplication 
  use some hardware tweeks or just improve the mem communication and manage the cache line 
  package all of it into a gemm.cu 

gemm.c -> learn how to use SIMD instructions !
  use AVX256 to time the matrix multiplication 
  after that publish both AVX256 and 512 functionality in the code 

Document both the gemms and publish the results !!
