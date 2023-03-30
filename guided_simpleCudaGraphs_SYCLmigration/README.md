# `SimpleCudaGraphs` Sample

## Prior knowledge

- [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - Beginner
- [SYCL](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) - Beginner
- [SYCLomatic Manual](https://github.com/oneapi-src/SYCLomatic#syclomatic)

## Prerequisites

| Property              | Description
|:---                   |:---
| OS                    | Ubuntu* 20.04
| Hardware              | Skylake with GEN9 or newer
| Software              | Intel® oneAPI DPC++/C++ Compiler

## Source code

- [CUDA](https://github.com/NVIDIA/cuda-samples/tree/v11.8/Samples/3_CUDA_Features/simpleCudaGraphs) - Source code 
- [SYCL](https://github.com/ShwethaSelma/simpleCudaGraphs/tree/master/guided_simpleCudaGraphs_SYCLmigration) - Migrated Code

## Purpose

The sample shows the migration of simple explicit CUDA Graph API's such as cudaGraphCreate, cudaGraphAddMemcpyNode, cudaGraphClone etc, to SYCL equivalent API's using [Taskflow](https://github.com/taskflow/taskflow) programming Model. The parallel implementation demonstrates the use of CUDA Graph API's, CUDA streams, shared memory, cooperative groups and warp level primitives. 

This sample contains two versions in the following folders:

| Folder Name                   | Description
|:---                           |:---
| `01_dpct_output`              | Contains output of Intel® SYCLomatic Compatibility Tool used to migrate SYCL-compliant code from CUDA code. This SYCL code has some unmigrated code that has to be manually fixed to get full functionality. (The code does not functionally work as supplied.)
| `02_sycl_migrated`            | Contains manually migrated SYCL code from CUDA code.

## CUDA features demonstrated

This sample demonstrates the migration of the following prominent CUDA features: 
- CUDA Graph APIs
- CUDA Stream Capture
- Shared memory
- CUDA streams 
- Cooperative groups
- Warp level primitives

## CUDA source code evaluation

The simpleCudaGraphs sample demonstrates the usage of CUDA Graphs API’s by performing element reduction. The CUDA Graph API are demonstrated in two CUDA functions cudaGraphsManual() which uses explicit CUDA Graph APIs and cudaGraphsUsingStreamCapture() which uses stream capture APIs. Reduction is performed in two CUDA kernels reduce () and reduceFinal(). We only migrate the cudaGraphsManual() using SYCLomatic Tool and manually migrating the unmigrated code section using [Taskflow](https://github.com/taskflow/taskflow) Programming Model. We do not migrate cudaGraphsUsingStreamCapture() because CUDA Stream Capture APIs are not yet supported in SYCL.

## Workflow For CUDA to SYCL migration

Refer [Workflow](https://www.intel.com/content/www/us/en/developer/tools/oneapi/training/cuda-sycl-migration-workflow.html#gs.s2njvh) for details.

## Set Environment Variables

When working with the command-line interface (CLI), you should configure the oneAPI toolkits using environment variables. Set up your CLI environment by sourcing the `setvars` script every time you open a new terminal window. This practice ensures that your compiler, libraries, and tools are ready for development.

## Tool assisted migration – SYCLomatic 

For this sample, the Intel SYCLomatic Compatibility tool automatically migrates ~80% of the CUDA code to SYCL. Follow these steps to generate the SYCL code using the compatibility tool:

1. git clone https://github.com/NVIDIA/cuda-samples.git
2. cd cuda-samples/Samples/3_CUDA_Features/simpleCudaGraphs/
3. Generate a compilation database with intercept-build
   ```
   intercept-build make
   ```
4. The above step creates a JSON file named compile_commands.json with all the compiler invocations and stores the names of the input files and the compiler options.
5. Pass the JSON file as input to the Intel SYCLomatic Compatibility Tool. The result is written to a folder named dpct_output. The --in-root specifies path to the root of the source tree to be migrated.
   ```
   c2s -p compile_commands.json --in-root ../../..
   ```
   
## Manual workarounds 
   

## Build the `simpleCudaGraphs` Sample for CPU and GPU

> **Note**: If you have not already done so, set up your CLI
> environment by sourcing  the `setvars` script in the root of your oneAPI installation.
>
> Linux*:
> - For system wide installations: `. /opt/intel/oneapi/setvars.sh`
> - For private installations: ` . ~/intel/oneapi/setvars.sh`
> - For non-POSIX shells, like csh, use the following command: `bash -c 'source <install-dir>/setvars.sh ; exec csh'`
>
> For more information on configuring environment variables, see [Use the setvars Script with Linux* or macOS*](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/oneapi-development-environment-setup/use-the-setvars-script-with-linux-or-macos.html).

### On Linux*

1. Change to the sample directory.
2. Build the program.
   ```
   $ mkdir build
   $ cd build
   $ cmake ..
   $ make
   ```

   By default, this command sequence will build the `02_sycl_migrated` versions of the program.

#### Troubleshooting

If an error occurs, you can get more details by running `make` with
the `VERBOSE=1` argument:
```
make VERBOSE=1
```
If you receive an error message, troubleshoot the problem using the **Diagnostics Utility for Intel® oneAPI Toolkits**. The diagnostic utility provides configuration and system checks to help find missing dependencies, permissions errors, and other issues. See the [Diagnostics Utility for Intel® oneAPI Toolkits User Guide](https://www.intel.com/content/www/us/en/develop/documentation/diagnostic-utility-user-guide/top.html) for more information on using the utility.


## Run the `simpleCudaGraphs` Sample

### On Linux

You can run the programs for CPU and GPU. The commands indicate the device target.

1. Run `02_sycl_migrated` for CPU and GPU.
    ```
    make run_cpu
    make run_gpu
    ```

### Example Output

The following example is for `02_sycl_migrated` for GPU on **Intel(R) UHD Graphics [0x9a60]**.
```
16777216 elements
threads per block  = 512
Graph Launch iterations = 3
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
[syclTaskFlowManual] Host callback final reduced sum = 0.996214

Number of tasks(nodes) in the syclTaskFlow(graph) created manually = 7
Cloned Graph Output.. 
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
[syclTaskFlowManual] Host callback final reduced sum = 0.996214
Built target run_gpu
```
>**Note**: On Gen11 architecture double data types are not supported, hence change double data types to float data types.

## License
Code samples are licensed under the MIT license. See
[License.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/License.txt) for details.

Third party program licenses are at [third-party-programs.txt](https://github.com/oneapi-src/oneAPI-samples/blob/master/third-party-programs.txt).

