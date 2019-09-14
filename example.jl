##############################################################################
# We include some examples of how to use the sandpile GPU function, 
#		GPU_stabilize(all4s_input, full_domain);
##############################################################################

include("sandpile_init.jl")

#### We include some examples of how to use the function 
####
#### 	output_sand, output_odometer = GPU_stabilize(input_sand, domain)
####
#### -------- 
#### INPUT VARIABLES: 
#### input_sand: the input sandpile, an N x N array of integers
#### domain : the domain on which to run the sandpile, an N x N array, 
####			where if domain[i,j] = -1,  then (i,j) is not in the domain
#### ---------
#### OUTPUT VARIABLES: 
#### output_sand: the stable sandpile, an N x N array of integers
#### output_odometer : the number of times each site topples when stabilizing, an N x N array. 



####################################
## single source sandpile
####################################
N = 100;
ss_input = zeros(N,N); center = Int(round(N/2)); ss_input[center,center]+=N^2
ss_sandpile, ss_odometer = GPU_stabilize(ss_input, full_domain)

####################################
## all 4s sandpile on a square domain
####################################
N = 100;
all4s_input = 4*ones(N,M);
full_domain = ones(N,M); 
all4s_sandpile, all4s_odometer = GPU_stabilize(all4s_input, full_domain);


####################################
## all 4s sandpile on a diamond domain
####################################

N = 200; 
diamond_domain = -1*ones(N,N); center = Int(round(N/2));
for i in 1:N 
	for j in 1:N
		x = i - center; y = j - center;
		if(norm([x; y],1) <= N/3)
			diamond_domain[i,j] = 1; 
		end
	end
end
all4s_input = 4*ones(N,N);
all4s_diamond, all4s_diamond_odometer = GPU_stabilize(all4s_input, diamond_domain)


