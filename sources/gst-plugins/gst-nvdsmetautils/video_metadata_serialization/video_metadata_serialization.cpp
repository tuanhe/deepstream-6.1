/**
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>

#include <gst/video/gstvideometa.h>
#include <gst/video/video.h>

#include "gstnvdsmeta.h"
#include "nvdscustomusermeta.h"

#define NVDS_VIDEO_METADATA 0xABCDEF

GST_DEBUG_CATEGORY_STATIC (gst_serialization_debug_category);
#define GST_CAT_DEFAULT gst_serialization_debug_category

void *set_metadata_ptr(void);
static gpointer copy_user_meta(gpointer data, gpointer user_data);
static void release_user_meta(gpointer data, gpointer user_data);

void *set_metadata_ptr(void *mem, guint mem_size)
{
    NVDS_CUSTOM_PAYLOAD *metadata = (NVDS_CUSTOM_PAYLOAD *)g_malloc0(sizeof (NVDS_CUSTOM_PAYLOAD));

    metadata->payloadType = NVDS_VIDEO_METADATA;
    metadata->payloadSize = mem_size;
    metadata->payload     = (uint8_t *) mem;

    return (void *) metadata;
}

static gpointer copy_user_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NVDS_CUSTOM_PAYLOAD *src_user_metadata = (NVDS_CUSTOM_PAYLOAD *)user_meta->user_meta_data;
    NVDS_CUSTOM_PAYLOAD *dst_user_metadata = (NVDS_CUSTOM_PAYLOAD *)g_malloc0(sizeof(NVDS_CUSTOM_PAYLOAD));
    dst_user_metadata->payloadType         = src_user_metadata->payloadType;
    dst_user_metadata->payloadSize         = src_user_metadata->payloadSize;
    dst_user_metadata->payload             = (uint8_t *)g_malloc0(src_user_metadata->payloadSize);
    memcpy(dst_user_metadata->payload, src_user_metadata->payload, src_user_metadata->payloadSize * sizeof (uint8_t));
    return (gpointer)dst_user_metadata;
}

static void release_user_meta(gpointer data, gpointer user_data)
{
    NvDsUserMeta *user_meta = (NvDsUserMeta *)data;
    NVDS_CUSTOM_PAYLOAD *user_metadata = (NVDS_CUSTOM_PAYLOAD *)user_meta->user_meta_data;
    g_free (user_metadata->payload);
    user_metadata->payload = NULL;
    g_free (user_meta->user_meta_data);
    user_meta->user_meta_data = NULL;
    return;
}

extern "C" void serialize_data (GstBuffer *buf);
void serialize_data (GstBuffer *buf)
{
    GST_DEBUG_CATEGORY_INIT (gst_serialization_debug_category,
        "serialization", 0, "serialization");
    void *mem = NULL;
    guint mem_size = 0;
    NvDsUserMeta *user_meta = NULL;
    NvDsMetaType user_meta_type = NVDS_USER_CUSTOM_META;
    {
        NvDsMetaList * l_frame = NULL;
        NvDsMetaList * l_obj = NULL;
        NvDsObjectMeta *obj_meta = NULL;
        NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

        if (batch_meta)
        {
            for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
            {
                NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

                mem_size += sizeof (NvDsFrameMeta);
                mem_size += frame_meta->num_obj_meta * sizeof (NvDsObjectMeta);
                void *mem = g_malloc0(mem_size);
                uint8_t* mem_ptr = (uint8_t *)mem;

                GST_DEBUG ("SERIALIZE ============= payload size = %d , objects in this frame = %d , width = %d , height = %d\n",
                        mem_size, frame_meta->num_obj_meta, frame_meta->source_frame_width, frame_meta->source_frame_height);

                /*
                 * Note:
                 * serialization is currently done for framemeta and objectmeta inside it,
                 * rest of the metadata are not serialized yet.
                 * */

                NvDsFrameMeta *serialized_frame_meta = (NvDsFrameMeta *) mem_ptr;
                serialized_frame_meta->pad_index           = frame_meta->pad_index;
                serialized_frame_meta->batch_id            = frame_meta->batch_id;
                serialized_frame_meta->frame_num           = frame_meta->frame_num;
                serialized_frame_meta->buf_pts             = frame_meta->buf_pts;
                serialized_frame_meta->ntp_timestamp       = frame_meta->ntp_timestamp;
                serialized_frame_meta->num_surfaces_per_frame = frame_meta->num_surfaces_per_frame;
                serialized_frame_meta->source_id           = frame_meta->source_id;
                serialized_frame_meta->source_frame_width  = frame_meta->source_frame_width;
                serialized_frame_meta->source_frame_height = frame_meta->source_frame_height;
                serialized_frame_meta->surface_type        = frame_meta->surface_type;
                serialized_frame_meta->surface_index       = frame_meta->surface_index;
                serialized_frame_meta->num_obj_meta        = frame_meta->num_obj_meta;
                serialized_frame_meta->bInferDone          = frame_meta->bInferDone;

                NvDsObjectMeta *serialized_object_meta = (NvDsObjectMeta *)((uint8_t *)serialized_frame_meta + sizeof (NvDsFrameMeta));

                for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
                {
                    obj_meta = (NvDsObjectMeta *) (l_obj->data);
                    serialized_object_meta->tracker_confidence = obj_meta->tracker_confidence;
                    serialized_object_meta->confidence         = obj_meta->confidence;
                    serialized_object_meta->class_id           = obj_meta->class_id;
                    serialized_object_meta->object_id          = obj_meta->object_id;
                    serialized_object_meta->detector_bbox_info = obj_meta->detector_bbox_info;
                    serialized_object_meta->rect_params        = obj_meta->rect_params;
                    serialized_object_meta->mask_params        = obj_meta->mask_params;
                    serialized_object_meta->text_params        = obj_meta->text_params;
                    memcpy (serialized_object_meta->obj_label, obj_meta->obj_label, MAX_LABEL_SIZE * sizeof (gchar));
                    memcpy (serialized_object_meta->misc_obj_info, obj_meta->misc_obj_info, MAX_USER_FIELDS * sizeof (gint64));
                    memcpy (serialized_object_meta->reserved, obj_meta->reserved, MAX_RESERVED_FIELDS * sizeof (gint64));

                    serialized_object_meta++;
                }

                user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

                user_meta->user_meta_data         = (void *)set_metadata_ptr(mem, mem_size);
                user_meta->base_meta.meta_type    = user_meta_type;
                user_meta->base_meta.copy_func    = (NvDsMetaCopyFunc)copy_user_meta;
                user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_user_meta;

                /* We want to add NvDsUserMeta to frame level */
                nvds_add_user_meta_to_frame(frame_meta, user_meta);

                mem_size = 0;
            }
        }
    }
}

extern "C" void deserialize_data (GstBuffer *buf);
void deserialize_data (GstBuffer *buf)
{
    GST_DEBUG_CATEGORY_INIT (gst_serialization_debug_category,
        "deserialization", 0, "deserialization");
    NvDsMetaList * l_frame         = NULL;
    NvDsUserMeta *user_meta        = NULL;
    NVDS_CUSTOM_PAYLOAD *metadata = NULL;
    NvDsMetaList * l_user_meta     = NULL;
    NvDsMetaList * l_obj = NULL;
    NvDsObjectMeta *obj_meta = NULL;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);

    if (batch_meta == NULL)
      return;

    const gchar *clear_nvds_batch_meta = g_getenv ("CLEAR_NVDS_BATCH_META");

    if (clear_nvds_batch_meta != NULL && !strcmp(clear_nvds_batch_meta, "yes"))
    {
        for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
        {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
            for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next)
            {
                obj_meta = (NvDsObjectMeta *) (l_obj->data);
                memset (&obj_meta->rect_params, 0, sizeof (NvOSD_RectParams));
            }

            frame_meta->pad_index           =
                frame_meta->batch_id            =
                frame_meta->frame_num           =
                frame_meta->ntp_timestamp       =
                frame_meta->num_surfaces_per_frame =
                frame_meta->source_id           =
                frame_meta->source_frame_width  =
                frame_meta->source_frame_height =
                frame_meta->surface_type        =
                frame_meta->surface_index       =
                frame_meta->num_obj_meta        =
                frame_meta->bInferDone          = 0;
        }
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);

        for (l_user_meta = frame_meta->frame_user_meta_list; l_user_meta != NULL; l_user_meta = l_user_meta->next)
        {
            user_meta = (NvDsUserMeta *) (l_user_meta->data);
            if(user_meta->base_meta.meta_type == NVDS_USER_CUSTOM_META)
            {
                metadata = (NVDS_CUSTOM_PAYLOAD *)user_meta->user_meta_data;
                NvDsFrameMeta *fmeta = (NvDsFrameMeta *)metadata->payload;
                if (metadata->payloadType == NVDS_VIDEO_METADATA)
                {

                    frame_meta->pad_index           = fmeta->pad_index;
                    frame_meta->batch_id            = fmeta->batch_id;
                    frame_meta->frame_num           = fmeta->frame_num;
                    frame_meta->buf_pts             = fmeta->buf_pts;
                    frame_meta->ntp_timestamp       = fmeta->ntp_timestamp;
                    frame_meta->num_surfaces_per_frame = fmeta->num_surfaces_per_frame;
                    frame_meta->source_id           = fmeta->source_id;
                    frame_meta->source_frame_width  = fmeta->source_frame_width;
                    frame_meta->source_frame_height = fmeta->source_frame_height;
                    frame_meta->surface_type        = fmeta->surface_type;
                    frame_meta->surface_index       = fmeta->surface_index;
                    frame_meta->num_obj_meta        = 0;//fmeta->num_obj_meta;
                    frame_meta->bInferDone          = fmeta->bInferDone;

                    GST_DEBUG ("DE-SERIALIZE ^^^^^^^^^^ payload size = %d , objects in this frame = %d . width = %d . height = %d\n",
                            metadata->payloadSize, fmeta->num_obj_meta, fmeta->source_frame_width, fmeta->source_frame_height);
                }

                NvDsObjectMeta *tobjmeta = (NvDsObjectMeta *)((uint8_t *)fmeta + sizeof (NvDsFrameMeta));
                for (int i=0; i < fmeta->num_obj_meta; i++)
                {
                    obj_meta = nvds_acquire_obj_meta_from_pool (batch_meta);
                    obj_meta->unique_component_id = tobjmeta->unique_component_id;
                    obj_meta->confidence = tobjmeta->confidence;

                    /* This is an untracked object. Set tracking_id to -1. */
                    obj_meta->object_id = tobjmeta->object_id;
                    obj_meta->class_id = tobjmeta->class_id;

                    obj_meta->rect_params = tobjmeta->rect_params;

                    nvds_add_obj_meta_to_frame (frame_meta, obj_meta, NULL);
                    tobjmeta++;
                }

            }
        }
    }
}
