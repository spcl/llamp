/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#ifndef CH4_REQUEST_H_INCLUDED
#define CH4_REQUEST_H_INCLUDED

#include "ch4_impl.h"
#include "mpidu_genq.h"
#include "time.h"

MPL_STATIC_INLINE_PREFIX int MPID_Request_is_anysource(MPIR_Request * req)
{
    MPIR_FUNC_ENTER;

    MPIR_Assert(0);

    MPIR_FUNC_EXIT;
    return MPI_SUCCESS;
}

MPL_STATIC_INLINE_PREFIX int MPID_Request_is_pending_failure(MPIR_Request * req)
{
    MPIR_FUNC_ENTER;

    MPIR_Assert(0);

    MPIR_FUNC_EXIT;
    return MPI_SUCCESS;
}

MPL_STATIC_INLINE_PREFIX void MPID_Request_set_completed(MPIR_Request * req)
{
    MPIR_FUNC_ENTER;

    MPIR_cc_set(&req->cc, 0);

    MPIR_FUNC_EXIT;
    return;
}

/* These request functions should be called by the MPI layer only
   since they only do base initialization of the request object.
   A few notes:

   It is each layer's responsibility to initialize a request
   properly.

   The CH4I_request functions are even more bare bones.
   They create request objects that are not usable by the
   lower layers until further initialization takes place.

   MPIDIG_request_xxx functions can be used to create and destroy
   request objects at any CH4 layer, including shmmod and netmod.
   These functions create and initialize a base request with
   the appropriate "above device" fields initialized, and any
   required CH4 layer fields initialized.

   The net/shm mods can upcall to MPIDIG to create a request, or
   they can iniitalize their own requests internally, but note
   that all the fields from the upper layers must be initialized
   properly.

   Note that the request_release function is used by the MPI
   layer to release the ref on a request object.  It is important
   for the netmods to release any memory pointed to by the request
   when the internal completion counters hits zero, NOT when the
   ref hits zero or there will be a memory leak. The generic
   release function will not release any memory pointed to by
   the request because it does not know about the internals of
   the mpidig/netmod/shmmod fields of the request.
*/
MPL_STATIC_INLINE_PREFIX int MPID_Request_complete_impl(MPIR_Request * req)
{
    int incomplete;

    MPIR_FUNC_ENTER;

    MPIR_cc_decr(req->cc_ptr, &incomplete);

    /* if we hit a zero completion count, free up AM-related
     * objects */
    if (!incomplete) {
        if (req->dev.completion_notification) {
            MPIR_cc_dec(req->dev.completion_notification);
        }

        if (req->dev.type == MPIDI_REQ_TYPE_AM) {
            /* FIXME: refactor mpidig code into ch4r_request.h */
            int vci = MPIDI_Request_get_vci(req);
            MPIDU_genq_private_pool_free_cell(MPIDI_global.per_vci[vci].request_pool,
                                              MPIDIG_REQUEST(req, req));
            MPIDIG_REQUEST(req, req) = NULL;
            MPIDI_NM_am_request_finalize(req);
#ifndef MPIDI_CH4_DIRECT_NETMOD
            MPIDI_SHM_am_request_finalize(req);
#endif
        }
        MPIDI_CH4_REQUEST_FREE(req);
    }

    MPIR_FUNC_EXIT;
    return MPI_SUCCESS;
}

MPL_STATIC_INLINE_PREFIX void MPIDI_Request_complete_fast_impl(MPIR_Request * req)
{
    int incomplete;
    MPIR_cc_decr(req->cc_ptr, &incomplete);
    if (!incomplete) {
        MPIDI_CH4_REQUEST_FREE(req);
    }
}


/**
 * Pushes the given request to the queue, and adds the timestamp
 * which indicates when the request should be dequeued and completed.
 */
static inline void enqueue_req(MPIR_Request * rreq, bool is_fast_complete)
{
    // Enqueues the request
    struct req_wrapper *delay_req = &delay_queue[delay_queue_write];
    // Only enqueues RECV requests
    if (delay > 0 && rreq->kind == MPIR_REQUEST_KIND__RECV) {
      // Adds the end timestamp
      clock_gettime(CLOCK_MONOTONIC, &delay_req->end);
      delay_req->end.tv_nsec += delay * 1000;
      delay_req->end.tv_sec += delay_req->end.tv_nsec / 1000000000;
      delay_req->end.tv_nsec %= 1000000000;
      
      /* printf("[DEBUG] Adding request to queue: %d [%ld:%ld]\n", delay_queue_write,  */
      /* 	     delay_req->end.tv_sec, delay_req->end.tv_nsec); */
      // Increments the write counter
      delay_queue_write = (delay_queue_write + 1) % MAX_QUEUE_SIZE;

      // Makes sure that the buffer is not overflown
      queue_size++;
      MPIR_Assert(queue_size <= MAX_QUEUE_SIZE);

      delay_req->is_fast_complete = is_fast_complete;
      // Request pointer needs to be added last
      delay_req->req = rreq;
    }
    else
    {
      // Makes sure that the buffer is not overflown
      queue_size++;
      MPIR_Assert(queue_size <= MAX_QUEUE_SIZE);
      // Enqueues the request directly
      delay_queue_write = (delay_queue_write + 1) % MAX_QUEUE_SIZE;
      delay_req->req = rreq;
    }
}


MPL_STATIC_INLINE_PREFIX int MPID_Request_complete(MPIR_Request * req)
{
  enqueue_req(req, false);
}


MPL_STATIC_INLINE_PREFIX void MPIDI_Request_complete_fast(MPIR_Request * req)
{
  enqueue_req(req, true);
}


MPL_STATIC_INLINE_PREFIX void MPID_Prequest_free_hook(MPIR_Request * req)
{
    MPIR_FUNC_ENTER;

    /* If a user passed a derived datatype for this persistent communication,
     * free it.
     * We could have done this cleanup in more general request cleanup functions,
     * like MPID_Request_destroy_hook. However, that would always add a few
     * instructions for any kind of request object, even if it's no a request
     * from persistent communications. */
    MPIR_Datatype_release_if_not_builtin(MPIDI_PREQUEST(req, datatype));

    MPIR_FUNC_EXIT;
}

MPL_STATIC_INLINE_PREFIX void MPID_Part_request_free_hook(MPIR_Request * req)
{
    MPIR_FUNC_ENTER;

    MPIR_Datatype_release_if_not_builtin(MPIDI_PART_REQUEST(req, datatype));

    MPIR_FUNC_EXIT;
}



#endif /* CH4_REQUEST_H_INCLUDED */
