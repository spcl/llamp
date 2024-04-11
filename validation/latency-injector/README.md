### Latency Injector in MPICH

##### Quick start
To simulate network latency in MPI with our implementation of the latency injector, first download __MPICH 4.1.2__ and __UCX 1.16.x__. Then, enter the following commands:
```console
> cp mpich-src/mpir_request.h <mpich-root>/src/include/
> cp mpich-src/ucx_recv.h <mpich-root>/src/mpid/ch4/netmod/ucx/
> cp mpich-src/ch4_impl.h ch4_init.c ch4_request.h <mpich-root>/src/mpid/ch4/src/
> cp ucx-src/ucp.h <ucx-root>/src/ucp/api/
> cp ucx-src/ucp_request.h ucx-src/ucp_request.inl <mpich-root>/src/ucp/core/
> cp ucx-src/tag_match.c ucx-src/tag_match.inl ucx-src/tag_recv.c <mpich-root>/src/ucp/tag/
```
After configuring and building MPICH and UCX, you can now use the environment variable `INJECTED_LATENCY` to specify the amount of delay (in microseconds) to be added to each p2p operations (i.e., sends and recvs), including the ones used in collectives.

Note that the maximum value of `INJECTED_LATENCY` is __1 second__ or __1000000 microseconds__. This value can be adjusted inside `ch4_impl.h` file. In addition, you might run into segmentation faults or see the assertion fail due to reaching the `MAX_QUEUE_SIZE`, which is currently set to 500 in `ch4_impl.h`. Since we do not have a queue whose size dynamically adjusts, this might happen sometimes, and the quickest way to deal with it is to simply recompile MPICH with a larger value of `MAX_QUEUE_SIZE`.

The general idea of this latency injector is that it intercepts the completion of RECV requests, stores them temporarily in a queue (i.e., a ring buffer) by the progress thread. The requests are stored along with timestamps that indicate when they can be released back to the main thread as per the injected latency. The queue is constantly polled by a separate request handler thread, which always checks the front of the queue, and once the current time reaches a request's `end` timestamp, the request will be released by calling the `complete` functions inside `ch4_request.h`. More details of the latency injector are described in the paper.

