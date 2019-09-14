using OpenCL


# C code to run OpenCL
const kernel_source = """
__kernel void topple(const int Ndim, __global float* A, __global float* B, __global float* toppleA)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
  
    float tmp = 0;  
    float curr = B[(i+1)*(Ndim+2) +j+1]; 
    float down = B[(i)*(Ndim+2) +j+1]; 
    float up = B[(i+2)*(Ndim+2) +j+1]; 
    float left = B[(i+1)*(Ndim+2) +j]; 
    float right =B[(i+1)*(Ndim+2) +j+2]; 

    tmp-=4*floor(curr/4); // topple current
    tmp+=floor(down/4); // down neighbor
    tmp+=floor(right/4); // right neighbor
    tmp+=floor(left/4); // left neighbor
    tmp+=floor(up/4); // up neighbor


    A[(i+1)*(Ndim+2) + j+1]+=tmp;
    toppleA[(i+1)*(Ndim+2) + j+1]+=floor(curr/4); 
}

__kernel void update(const int Ndim, __global float* shape,  __global float* A, __global float* B, __global float* C)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float before = B[(i+1)*(Ndim+2) + j+1];
    float shape_num = shape[(i+1)*(Ndim+2) + j+1];
    float after = A[(i+1)*(Ndim+2) + j+1];

    if(shape_num == -1){ 
        after = 0;
    }
    if(before != after){
        C[0]=1;
    }
    
    B[(i+1)*(Ndim+2) + j+1]=after; 
}


""";


function GPU_stabilize(sandpile, shape, ctx = 0)
    N,M = size(sandpile); 


    #pad sandpile so that everything is always inbounds
    sandpile_vec = zeros((N+2)*(M+2)); 
    topple_vec = zeros((N+2)*(M+2)); 
    shape_vec = zeros((N+2)*(M+2)); 
    for i in 1:N
        for j in 1:M
            sandpile_vec[(i)*(N+2) + j+1]=sandpile[i,j]
            shape_vec[(i)*(N+2) + j+1]=shape[i,j]
        end
    end

    #convert to FLOAT32
    sandpile_vec = convert(Array{Float32,1}, sandpile_vec);
    shape_vec = convert(Array{Float32,1}, shape_vec);
    topple_vec = convert(Array{Float32,1}, topple_vec);
    
    # set up OpenCL
    if(ctx == 0)
        ctx = cl.create_some_context()
    end
    queue = cl.CmdQueue(ctx)
    


    #set up variables
    h_C = Vector{Float32}(1) 
    d_a = cl.Buffer(Float32, ctx, (:r,:copy), hostbuf=sandpile_vec);
    d_b = cl.Buffer(Float32, ctx, (:r,:copy), hostbuf=sandpile_vec);
    d_shape = cl.Buffer(Float32, ctx, (:r,:copy), hostbuf=shape_vec);
    d_aa = cl.Buffer(Float32, ctx, (:r,:copy), hostbuf=topple_vec);

    #start the openCL program
    prg  = cl.Program(ctx, source=kernel_source) |> cl.build!
    topple_C = cl.Kernel(prg, "topple")
    update_C = cl.Kernel(prg, "update");


    #run the sandpile till completion
    for i in 1:(N^2*M^2);
        h_C[1] = 0; 
        d_c = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf = h_C);
        global_range = (N, M)
        topple_ocl = topple_C[queue, global_range];
        update_ocl = update_C[queue, global_range]
        topple_ocl(Int32(round(N)),d_a, d_b,d_aa);
        update_ocl(Int32(round(N)),d_shape, d_a, d_b, d_c);
        cl.copy!(queue, h_C, d_c);
        if(h_C[1] == 0)
            break;
        end
        
    end   
    
    cl.copy!(queue, sandpile_vec, d_b);
    cl.copy!(queue, topple_vec, d_aa); 
    sandpile = reshape(sandpile_vec, N+2,M+2); 
    topple = reshape(topple_vec, N+2,M+2); 
    return sandpile[2:N+1,2:M+1]', topple[2:N+1, 2:M+1]'; 
end

