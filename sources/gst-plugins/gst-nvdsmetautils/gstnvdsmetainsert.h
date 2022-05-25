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

#ifndef __GST_NVDSMETAINSERT_H__
#define __GST_NVDSMETAINSERT_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>

G_BEGIN_DECLS

/* #defines don't like whitespacey bits */
#define GST_TYPE_NVDSMETAINSERT \
  (gst_nvdsmetainsert_get_type())
#define GST_NVDSMETAINSERT(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVDSMETAINSERT,Gstnvdsmetainsert))
#define GST_NVDSMETAINSERT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVDSMETAINSERT,GstnvdsmetainsertClass))
#define GST_IS_NVDSMETAINSERT(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVDSMETAINSERT))
#define GST_IS_NVDSMETAINSERT_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVDSMETAINSERT))

typedef struct _Gstnvdsmetainsert      Gstnvdsmetainsert;
typedef struct _GstnvdsmetainsertClass GstnvdsmetainsertClass;

struct _Gstnvdsmetainsert
{
  GstBaseTransform element;

  GstPad *sinkpad, *srcpad;
  gboolean is_same_caps;

  /* source and sink pad caps */
  GstCaps *sinkcaps;
  GstCaps *srccaps;

  gchar* serialization_lib_name;
  void *lib_handle;
  void (*serialize_func)(GstBuffer *buf);
  guint meta_mem_size;
  void *meta_mem;

};

struct _GstnvdsmetainsertClass
{
  GstBaseTransformClass parent_class;
};

GType gst_nvdsmetainsert_get_type (void);

gboolean nvds_metainsert_init (GstPlugin * nvdsmetainsert);

G_END_DECLS

#endif /* __GST_NVDSMETAINSERT_H__ */
