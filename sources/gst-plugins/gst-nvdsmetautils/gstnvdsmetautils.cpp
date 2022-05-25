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

#include "gstnvdsmetainsert.h"
#include "gstnvdsmetaextract.h"

#define PACKAGE_LICENSE     "Proprietary"
#define PACKAGE_NAME        "GStreamer NV DS META Data Processor Plugins"
#define PACKAGE_URL         "http://nvidia.com/"
#define PACKAGE_DESCRIPTION "DS Elements for META insertion & extraction"

#ifndef PACKAGE
#define PACKAGE "nvdsmetautils"
#endif

static gboolean plugin_init (GstPlugin * plugin)
{
    gboolean ret = TRUE;
    nvds_metainsert_init (plugin);
    nvds_metaextract_init (plugin);
    return ret;
}


GST_PLUGIN_DEFINE (
        GST_VERSION_MAJOR,
        GST_VERSION_MINOR,
        nvdsgst_metautils,
        PACKAGE_DESCRIPTION,
        plugin_init,
        "6.1",
        PACKAGE_LICENSE,
        PACKAGE_NAME,
        PACKAGE_URL)
