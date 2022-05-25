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

#define NVDS_AUDIO_METADATA 0xFEDCBA

GST_DEBUG_CATEGORY_STATIC (gst_audio_serialization_debug_category);
#define GST_CAT_DEFAULT gst_audio_serialization_debug_category

void *set_audio_metadata_ptr(void);
static gpointer copy_audio_user_meta(gpointer data, gpointer user_data);
static void release_audio_user_meta(gpointer data, gpointer user_data);

void *set_audio_metadata_ptr(void *mem, guint mem_size)
{
    NVDS_CUSTOM_PAYLOAD *metadata = (NVDS_CUSTOM_PAYLOAD *)g_malloc0(sizeof (NVDS_CUSTOM_PAYLOAD));

    metadata->payloadType = NVDS_AUDIO_METADATA;
    metadata->payloadSize = mem_size;
    metadata->payload     = (uint8_t *) mem;

    return (void *) metadata;
}

static gpointer copy_audio_user_meta(gpointer data, gpointer user_data)
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

static void release_audio_user_meta(gpointer data, gpointer user_data)
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
    GST_DEBUG_CATEGORY_INIT (gst_audio_serialization_debug_category,
        "audioserialization", 0, "audio serialization");
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
                NvDsAudioFrameMeta *frame_meta = (NvDsAudioFrameMeta *) (l_frame->data);

                mem_size += sizeof (NvDsAudioFrameMeta);
                void *mem = g_malloc0(mem_size);
                uint8_t* mem_ptr = (uint8_t *)mem;

                GST_DEBUG ("SERIALIZE ============= payload size = %d , samplerate = %d layout = %d \n",
                        mem_size, frame_meta->sample_rate, frame_meta->layout);

                /*
                 * Note:
                 * serialization is currently done for framemeta and objectmeta inside it,
                 * rest of the metadata are not serialized yet.
                 * */

                NvDsAudioFrameMeta *serialized_frame_meta    = (NvDsAudioFrameMeta *) mem_ptr;
                serialized_frame_meta->pad_index             = frame_meta->pad_index;
                serialized_frame_meta->batch_id              = frame_meta->batch_id;
                serialized_frame_meta->frame_num             = frame_meta->frame_num;
                serialized_frame_meta->buf_pts               = frame_meta->buf_pts;
                serialized_frame_meta->ntp_timestamp         = frame_meta->ntp_timestamp;
                serialized_frame_meta->source_id             = frame_meta->source_id;
                serialized_frame_meta->num_samples_per_frame = frame_meta->num_samples_per_frame;
                serialized_frame_meta->sample_rate           = frame_meta->sample_rate;
                serialized_frame_meta->num_channels          = frame_meta->num_channels;
                serialized_frame_meta->format                = frame_meta->format;
                serialized_frame_meta->layout                = frame_meta->layout;
                serialized_frame_meta->bInferDone            = frame_meta->bInferDone;
                serialized_frame_meta->class_id              = frame_meta->class_id;
                serialized_frame_meta->confidence            = frame_meta->confidence;
                memcpy (serialized_frame_meta->class_label, frame_meta->class_label, MAX_LABEL_SIZE * sizeof (gchar));

                user_meta = nvds_acquire_user_meta_from_pool(batch_meta);

                user_meta->user_meta_data         = (void *)set_audio_metadata_ptr(mem, mem_size);
                user_meta->base_meta.meta_type    = user_meta_type;
                user_meta->base_meta.copy_func    = (NvDsMetaCopyFunc)copy_audio_user_meta;
                user_meta->base_meta.release_func = (NvDsMetaReleaseFunc)release_audio_user_meta;

                /* We want to add NvDsUserMeta to frame level */
                nvds_add_user_meta_to_audio_frame(frame_meta, user_meta);

                mem_size = 0;
            }
        }
    }
}

extern "C" void deserialize_data (GstBuffer *buf);
void deserialize_data (GstBuffer *buf)
{
    GST_DEBUG_CATEGORY_INIT (gst_audio_serialization_debug_category,
        "audiodeserialization", 0, "audio deserialization");
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
            NvDsAudioFrameMeta *frame_meta = (NvDsAudioFrameMeta *) (l_frame->data);

            frame_meta->pad_index =
            frame_meta->batch_id =
            frame_meta->frame_num =
            frame_meta->buf_pts =
            frame_meta->ntp_timestamp =
            frame_meta->source_id =
            frame_meta->num_samples_per_frame =
            frame_meta->sample_rate =
            frame_meta->num_channels = 0;
            frame_meta->format = NVBUF_AUDIO_INVALID_FORMAT;
            frame_meta->layout = NVBUF_AUDIO_INVALID_LAYOUT;
            frame_meta->bInferDone =
            frame_meta->class_id =
            frame_meta->confidence = 0;
            frame_meta->class_label[MAX_LABEL_SIZE] = {};
        }
    }

    for (l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsAudioFrameMeta *frame_meta = (NvDsAudioFrameMeta *) (l_frame->data);

        for (l_user_meta = frame_meta->frame_user_meta_list; l_user_meta != NULL; l_user_meta = l_user_meta->next)
        {
            user_meta = (NvDsUserMeta *) (l_user_meta->data);
            if(user_meta->base_meta.meta_type == NVDS_USER_CUSTOM_META)
            {
                metadata = (NVDS_CUSTOM_PAYLOAD *)user_meta->user_meta_data;
                NvDsAudioFrameMeta *fmeta = (NvDsAudioFrameMeta *)metadata->payload;
                if (metadata->payloadType == NVDS_AUDIO_METADATA)
                {
                    frame_meta->pad_index           = fmeta->pad_index;
                    frame_meta->batch_id            = fmeta->batch_id;
                    frame_meta->frame_num           = fmeta->frame_num;
                    frame_meta->buf_pts             = fmeta->buf_pts;
                    frame_meta->ntp_timestamp       = fmeta->ntp_timestamp;
                    frame_meta->source_id           = fmeta->source_id;
                    frame_meta->num_samples_per_frame = fmeta->num_samples_per_frame;
                    frame_meta->sample_rate         = fmeta->sample_rate;
                    frame_meta->num_channels        = fmeta->num_channels;
                    frame_meta->format              = fmeta->format;
                    frame_meta->layout              = fmeta->layout;
                    frame_meta->bInferDone          = fmeta->bInferDone;
                    frame_meta->class_id            = fmeta->class_id;
                    frame_meta->confidence          = fmeta->confidence;
                    memcpy (frame_meta->class_label, fmeta->class_label, MAX_LABEL_SIZE * sizeof (gchar));

                    GST_DEBUG ("DE-SERIALIZE ^^^^^^^^^^ payload size = %d , frame num = %d , sample rate = %d , num channels = %d\n",
                            metadata->payloadSize, fmeta->frame_num, fmeta->sample_rate, fmeta->num_channels);
                }
            }
        }
    }
}
