/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __DEEPSTREAM_NMOS_APP_H__
#define __DEEPSTREAM_NMOS_APP_H__

#include <gst/gst.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define MAX_SOURCE_NUM  16
#define MAX_SINK_NUM    16

typedef enum NvDsNmosAppMode {
  NVDS_NMOS_APP_MODE_RECEIVE,
  NVDS_NMOS_APP_MODE_SEND,
  NVDS_NMOS_APP_MODE_RECVSEND
} NvDsNmosAppMode;

typedef enum NvDsNmosSrcType {
  NMOS_UDP_SRC_OSS = 1,
  NMOS_UDP_SRC_NV,
} NvDsNmosSrcType;

typedef enum NvDsNmosSinkType {
  NMOS_UDP_SINK_OSS = 1,
  NMOS_UDP_SINK_NV,
} NvDsNmosSinkType;

typedef struct NvDsNmosSrcConfig {
  gboolean enable;
  guint index;
  guint type;
  guint sinkType;
  guint packetsPerLine;
  guint payloadSize;
  gchar *sdpFile;
  gchar *sinkSdpFile;
  gchar *localIfaceIp;
  gchar *id;
  gchar *sinkId;
  gchar *srcSdpTxt;
  gchar *sinkSdpTxt;
} NvDsNmosSrcConfig;

typedef struct NvDsNmosSinkConfig {
  gboolean enable;
  guint index;
  guint type;
  guint packetsPerLine;
  guint payloadSize;
  gchar *sdpFile;
  gchar *localIfaceIp;
  gchar *id;
} NvDsNmosSinkConfig;

typedef struct NvDsNmosAppConfig {
  gboolean enablePgie;
  guint httpPort;
  guint numSrc;
  guint numSink;
  gchar *seed;
  gchar *hostName;
  gchar *pgieConfFile;
  NvDsNmosSrcConfig srcConfigs[MAX_SOURCE_NUM];
  NvDsNmosSinkConfig sinkConfigs[MAX_SINK_NUM];
} NvDsNmosAppConfig;

typedef struct NvDsNmosSrcBin
{
  GstElement *bin;
  GstElement *src;
  GstElement *queue;
  GstElement *payloader;
  GstElement *sink;
  gchar *mediaType;
  gchar *srcId;
  guint srcIndex;
} NvDsNmosSrcBin;

typedef struct NvDsNmosSinkBin
{
  GstElement *bin;
  GstElement *queue;
  GstElement *payloader;
  GstElement *sink;
  gchar *mediaType;
  gchar *id;
  guint index;
} NvDsNmosSinkBin;

typedef struct NvDsNmosAppCtx {
  GstElement *pipeline;
  guint watchId;
  gboolean isPipelineActive;
  GMainLoop *loop;
  NvDsNmosAppConfig config;
  GHashTable *sources;
  GHashTable *sinks;
} NvDsNmosAppCtx;

#ifdef __cplusplus
}
#endif

#endif