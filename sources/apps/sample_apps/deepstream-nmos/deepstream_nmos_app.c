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

#include <gst/gst.h>
#include <gst/sdp/sdp.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include "deepstream_common.h"
#include "deepstream_nmos_app.h"
#include "deepstream_nmos_config_parser.h"
#include "nvdsnmos.h"


typedef struct EventHandleData {
  GstElement *queue;
  GstElement *nextElem;
  GstElement *sink;
  gchar *ip;
  guint port;
} EventHandleData;

static gboolean cintr = FALSE;
static gchar *cfgFile = NULL;
static guint g_AppMode = 0;

GOptionEntry entries[] = {
  {"cfg-file", 'c', 0, G_OPTION_ARG_FILENAME, &cfgFile,
      "Set the config file", NULL},
  {"mode", 'm', 0, G_OPTION_ARG_INT, &g_AppMode,
      "App Mode; {0: Receive [DEFAULT]}, {1: Send}, {2: RecvSend}", NULL},
  {NULL},
};

/**
 * Function to handle program interrupt signal.
 * It installs default handler after handling the interrupt.
 */
static void
_intr_handler (int signum)
{
  struct sigaction action;

  NVGSTDS_ERR_MSG_V ("User Interrupted.. \n");

  memset (&action, 0, sizeof (action));
  action.sa_handler = SIG_DFL;

  sigaction (SIGINT, &action, NULL);

  cintr = TRUE;
}

/**
 * Loop function to check the status of interrupts.
 * It comes out of loop if application got interrupted.
 */
static gboolean
check_for_interrupt (gpointer data)
{
  if (cintr) {
    NvDsNmosAppCtx *appCtx = (NvDsNmosAppCtx *) data;
    cintr = FALSE;
    g_main_loop_quit (appCtx->loop);
    return FALSE;
  }
  return TRUE;
}

/*
 * Function to install custom handler for program interrupt signal.
 */
static void
_intr_setup (void)
{
  struct sigaction action;

  memset (&action, 0, sizeof (action));
  action.sa_handler = _intr_handler;

  sigaction (SIGINT, &action, NULL);
}

static gboolean
kbhit (void)
{
  struct timeval tv;
  fd_set rdfs;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  FD_ZERO (&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);

  select (STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET (STDIN_FILENO, &rdfs);
}

/**
 * Loop function to check keyboard inputs.
 */
static gboolean
event_thread_func (gpointer uData)
{
  NvDsNmosAppCtx *appCtx = (NvDsNmosAppCtx *) uData;
  gboolean ret = TRUE;

  // Check for keyboard input
  if (!kbhit ()) {
    return TRUE;
  }

  int c = fgetc (stdin);
  g_print ("\n");

  switch (c) {
    case 'q':
      g_main_loop_quit (appCtx->loop);
      ret = FALSE;
      break;
    default:
      break;
  }

  return ret;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  NvDsNmosAppCtx *appCtx = (NvDsNmosAppCtx *) data;
  GMainLoop *loop = (GMainLoop *) appCtx->loop;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_ERROR:{
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static const GstSDPMedia*
get_sdp_media (GstSDPMessage *sdpMsg, gchar *mediaType)
{
  g_return_val_if_fail (sdpMsg != NULL, NULL);
  g_return_val_if_fail (mediaType != NULL, NULL);

  guint mediaCount, i;
  mediaCount = gst_sdp_message_medias_len (sdpMsg);
  if (!mediaCount) {
    NVGSTDS_ERR_MSG_V ("No media found in sdp.");
    return NULL;
  }

  for (i = 0; i < mediaCount; i++) {
    const GstSDPMedia *media = gst_sdp_message_get_media (sdpMsg, i);
    if (!g_strcmp0 (media->media, mediaType))
      return media;
  }

  if (i == mediaCount)
    NVGSTDS_ERR_MSG_V ("No media of type (%s) found in sdp message", mediaType);

  return NULL;
}

static GstCaps*
get_video_caps_from_sdp_caps (GstCaps *srcCaps)
{
  g_return_val_if_fail (srcCaps != NULL, NULL);

  GstCaps *caps = NULL;
  const gchar *str;
  gint width, height, depth;
  gchar *format;

  GstStructure *structure = gst_caps_get_structure (srcCaps, 0);

  if (!(str = gst_structure_get_string (structure, "width"))) {
    NVGSTDS_ERR_MSG_V ("No width in sdp message");
    return NULL;
  }
  width = atoi (str);

  if (!(str = gst_structure_get_string(structure, "height"))) {
    NVGSTDS_ERR_MSG_V("No height in sdp message");
    return NULL;
  }
  height = atoi(str);

  if (!(str = gst_structure_get_string(structure, "depth")))
  {
    NVGSTDS_ERR_MSG_V("No depth in sdp message");
    return NULL;
  }
  depth = atoi (str);

  if (!(str = gst_structure_get_string (structure, "sampling"))) {
    NVGSTDS_ERR_MSG_V("No sampling in sdp message");
    return NULL;
  }

  if (!strcmp (str, "RGB")) {
    format = "RGB";
  } else if (!strcmp (str, "RGBA")) {
    format = "RGBA";
  } else if (!strcmp (str, "BGR")) {
    format = "BGR";
  } else if (!strcmp (str, "BGRA")) {
    format = "BGRA";
  } else if (!strcmp (str, "YCbCr-4:4:4")) {
    format = "AYUV";
  } else if (!strcmp (str, "YCbCr-4:2:2")) {
    if (depth == 8) {
      format = "UYVY";
    } else if (depth == 10) {
      format = "UYVP";
    } else {
      NVGSTDS_ERR_MSG_V ("Unknown sampling format in sdp message");
      return NULL;
    }
  } else if (!strcmp (str, "YCbCr-4:2:0")) {
    format = "I420";
  } else if (!strcmp (str, "YCbCr-4:1:1")) {
    format = "Y41B";
  } else {
    NVGSTDS_ERR_MSG_V ("Unknown sampling format in sdp message");
    return NULL;
  }

  caps = gst_caps_new_simple ("video/x-raw",
      "format", G_TYPE_STRING, format,
      "width", G_TYPE_INT, width,
      "height", G_TYPE_INT, height, NULL);

  return caps;
}

static GstCaps*
get_audio_caps_from_sdp_caps (GstCaps *srcCaps)
{
  g_return_val_if_fail (srcCaps != NULL, NULL);
  GstCaps *caps = NULL;
  gint channels, rate, tmp;
  const gchar *str;

  GstStructure *structure = gst_caps_get_structure (srcCaps, 0);

  if ((str = gst_structure_get_string (structure, "clock-rate"))) {
    rate = atoi(str);
  } else if (gst_structure_get_int (structure, "clock-rate", &tmp)) {
    rate = tmp;
  } else {
    NVGSTDS_ERR_MSG_V ("No clock-rate in sdp message");
    return NULL;
  }

  if ((str = gst_structure_get_string (structure, "encoding-params"))) {
    channels = atoi (str);
  } else if (gst_structure_get_int (structure, "channels", &tmp)) {
    channels = tmp;
  } else {
    NVGSTDS_INFO_MSG_V ("No channels details in sdp message, considering one channel");
    channels = 1;
  }

  caps = gst_caps_new_simple ("audio/x-raw",
      "format", G_TYPE_STRING, "S24BE",
      "rate", G_TYPE_INT, rate,
      "channels", G_TYPE_INT, channels, NULL);

  return caps;
}

static GstPadProbeReturn
event_probe_cb (GstPad * pad, GstPadProbeInfo * info, gpointer uData)
{
  EventHandleData *evData = (EventHandleData *) uData;

  if (GST_EVENT_TYPE (GST_PAD_PROBE_INFO_DATA (info)) != GST_EVENT_EOS)
    return GST_PAD_PROBE_PASS;

  gst_pad_remove_probe (pad, GST_PAD_PROBE_INFO_ID (info));

  if (gst_element_set_state (evData->sink,
          GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (evData->sink, "Can't set component to NULL");
    return GST_PAD_PROBE_DROP;
  }

  g_object_set (G_OBJECT (evData->sink), "host", evData->ip,
          "port", evData->port, NULL);

  if (!gst_element_sync_state_with_parent (evData->sink)) {
    GST_ERROR_OBJECT (evData->sink, "Couldn't sync state with parent");
    return GST_PAD_PROBE_DROP;
  }

  return GST_PAD_PROBE_DROP;
}

static GstPadProbeReturn
pad_probe_cb (GstPad * pad, GstPadProbeInfo * info, gpointer uData)
{
  GstPad *srcpad, *sinkpad;
  EventHandleData *evData = (EventHandleData *) uData;

  /* remove the probe first */
  gst_pad_remove_probe (pad, GST_PAD_PROBE_INFO_ID (info));

  /* install new probe for EOS */
  srcpad = gst_element_get_static_pad (evData->nextElem, "src");
  gst_pad_add_probe (srcpad, GST_PAD_PROBE_TYPE_BLOCK |
      GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM, event_probe_cb, uData, NULL);
  gst_object_unref (srcpad);

  /* push EOS into the element, the probe will be fired when the
   * EOS leaves the component and it has thus drained all of its data */
  sinkpad = gst_element_get_static_pad (evData->nextElem, "sink");
  gst_pad_send_event (sinkpad, gst_event_new_eos ());

  /* send the flush-start event to reset the EOS pad */
  gst_pad_send_event (sinkpad, gst_event_new_flush_start ());

  /* send the flush-stop to reset the flushing pad */
  gst_pad_send_event (sinkpad, gst_event_new_flush_stop (FALSE));
  gst_object_unref (sinkpad);

  g_free (evData->ip);
  g_free (evData);

  return GST_PAD_PROBE_OK;
}

static gboolean
update_active_component (GstElement *elemToUpdate, const gchar *sdpTxt,
    gboolean isSink, void *uData)
{
  g_return_val_if_fail (sdpTxt, FALSE);
  g_return_val_if_fail (elemToUpdate, FALSE);

  GstSDPResult result;
  GstSDPMessage *sdpMsg;
  result = gst_sdp_message_new_from_text (sdpTxt, &sdpMsg);
  if (result != GST_SDP_OK) {
    NVGSTDS_ERR_MSG_V ("Error (%d) in creating sdp message.", result);
    return FALSE;
  }

  const GstSDPMedia *media = gst_sdp_message_get_media (sdpMsg, 0);
  if (!media) {
    NVGSTDS_ERR_MSG_V ("Error!! No media in sdp message");
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  const GstSDPConnection *connection = gst_sdp_media_get_connection (media, 0);
  if (!connection) {
    NVGSTDS_ERR_MSG_V ("Error!! No connection info in sdp message");
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  NVGSTDS_INFO_MSG_V ("new connection addr:%s, port:%u\n", connection->address, media->port);

  if (isSink) {
    EventHandleData *evData = (EventHandleData *) uData;
    if (!evData) {
      NVGSTDS_ERR_MSG_V ("Invalid argument");
      return FALSE;
    }

    evData->ip = g_strdup (connection->address);
    evData->port = media->port;
    GstPad *srcpad = gst_element_get_static_pad (evData->queue, "src");
    // Add blocking probe to stop the data flow.
    gst_pad_add_probe (srcpad, GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
      pad_probe_cb, uData, NULL);

    gst_object_unref (srcpad);
    return TRUE;
  }

  if (gst_element_set_state (elemToUpdate,
          GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (elemToUpdate, "Can't set component to NULL");
    return FALSE;
  }

  g_object_set (G_OBJECT (elemToUpdate), "address", connection->address,
          "port", media->port, NULL);

  if (!gst_element_sync_state_with_parent (elemToUpdate)) {
    GST_ERROR_OBJECT (elemToUpdate, "Couldn't sync state with parent");
    return FALSE;
  }
  return TRUE;
}

static gpointer
create_audio_sender_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSinkConfig *sinkConfig,
    const GstSDPMedia *media)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (sinkConfig != NULL, FALSE);
  g_return_val_if_fail (media != NULL, FALSE);

  GstElement *sink = NULL;

  const GstSDPConnection *connection = gst_sdp_media_get_connection (media, 0);
  if (!connection) {
    NVGSTDS_ERR_MSG_V ("Error!! No connection info in sdp message");
    return NULL;
  }

  GstElement *audiobin = gst_bin_new ("nvds-audio-sender-bin");
  GstElement *source = gst_element_factory_make ("audiotestsrc", NULL);
  GstElement *capsfilter = gst_element_factory_make ("capsfilter", NULL);
  GstElement *queue = gst_element_factory_make ("queue", NULL);
  GstElement *payloader = gst_element_factory_make ("rtpL24pay", NULL);

  if (sinkConfig->type == NMOS_UDP_SINK_NV) {
    sink = gst_element_factory_make ("nvdsudpsink", NULL);
  } else {
    sink = gst_element_factory_make ("udpsink", NULL);
  }

  if (!source || !capsfilter || !payloader || !queue || !sink) {
    NVGSTDS_ERR_MSG_V ("Failed to create one of the component of audio sender pipeline.\n");
    return NULL;
  }

  gint pt = atoi (gst_sdp_media_get_format (media, 0));
  GstCaps *caps = gst_sdp_media_get_caps_from_media (media, pt);

  NVGSTDS_INFO_MSG_V ("received caps: %s\n", gst_caps_to_string (caps));
  if (caps) {
    GstCaps *acaps = get_audio_caps_from_sdp_caps (caps);
    if (acaps) {
      g_object_set (G_OBJECT(capsfilter), "caps", acaps, NULL);
      gst_caps_unref (acaps);
    } else {
      return NULL;
    }
  } else {
    NVGSTDS_ERR_MSG_V ("Failed to get caps from sdp message");
    return NULL;
  }
  gst_caps_unref (caps);

  if (sinkConfig->type == NMOS_UDP_SINK_NV) {
    const gchar *localIfaceIp = gst_sdp_media_get_attribute_val (media, "x-nvds-iface-ip");
    if (!localIfaceIp) {
      NVGSTDS_ERR_MSG_V ("No attribute x-nvds-iface-ip in sdp message");
      return FALSE;
    }
    g_object_set (G_OBJECT(sink), "local-iface-ip", localIfaceIp, "sdp-file",
        sinkConfig->sdpFile, NULL);

    const gchar *ptime = gst_sdp_media_get_attribute_val (media, "ptime");
    if (!ptime) {
      GST_ERROR ("No attribute ptime in sdp message");
      return FALSE;
    }

    gdouble tmpPtime = atof (ptime);
    if (tmpPtime <= 0) {
      GST_ERROR ("wrong value of ptime - %s", ptime);
      return FALSE;
    }
    guint64 ptime_ns = (guint64) (tmpPtime * 1000000 + 0.5);
    g_object_set (G_OBJECT(payloader), "max-ptime", ptime_ns, "ptime-multiple",
        ptime_ns, NULL);
  }

  g_object_set (G_OBJECT(source), "wave", 8, NULL);
  g_object_set (G_OBJECT(sink), "host", connection->address, "port",
      media->port, NULL);

  gst_bin_add_many (GST_BIN(audiobin), source, capsfilter, queue, payloader,
      sink, NULL);

  if (!gst_element_link_many (source, capsfilter, queue, payloader, sink, NULL)) {
    NVGSTDS_ERR_MSG_V ("Error in linking components of audio sender pipeline.");
    return NULL;
  }

  NvDsNmosSinkBin *sinkBin = g_new0 (NvDsNmosSinkBin, 1);
  sinkBin->bin = audiobin;
  sinkBin->sink = sink;
  sinkBin->queue = queue;
  sinkBin->payloader = payloader;
  sinkBin->mediaType = g_strdup ("audio");
  sinkBin->id = g_strdup (sinkConfig->id);

  return sinkBin;
}

static gpointer
create_video_sender_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSinkConfig *sinkConfig,
    const GstSDPMedia *media)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (sinkConfig != NULL, FALSE);
  g_return_val_if_fail (media != NULL, FALSE);

  GstElement *sink = NULL;

  const GstSDPConnection *connection = gst_sdp_media_get_connection (media, 0);
  if (!connection) {
    NVGSTDS_ERR_MSG_V ("Error!! No connection info in sdp message");
    return NULL;
  }

  GstElement *videobin = gst_bin_new ("nvds-video-bin");
  GstElement *source = gst_element_factory_make ("videotestsrc", NULL);
  GstElement *capsfilter = gst_element_factory_make ("capsfilter", NULL);
  GstElement *vparse = gst_element_factory_make ("rawvideoparse", NULL);
  GstElement *queue = gst_element_factory_make ("queue", NULL);
  GstElement *payloader = gst_element_factory_make ("rtpvrawpay", NULL);

  if (sinkConfig->type == NMOS_UDP_SINK_NV) {
    sink = gst_element_factory_make ("nvdsudpsink", NULL);
  } else {
    sink = gst_element_factory_make ("udpsink", NULL);
  }

  if (!source || !capsfilter || !payloader || !vparse || !queue || !sink) {
    NVGSTDS_ERR_MSG_V ("Failed to create one of the component of video sender pipeline.\n");
    return NULL;
  }

  gint pt = atoi (gst_sdp_media_get_format (media, 0));
  GstCaps *caps = gst_sdp_media_get_caps_from_media (media, pt);
  NVGSTDS_INFO_MSG_V ("received caps: %s\n", gst_caps_to_string (caps));
  if (caps) {
    GstCaps *vcaps = get_video_caps_from_sdp_caps (caps);
    if (vcaps) {
      g_object_set (G_OBJECT(capsfilter), "caps", vcaps, NULL);
      gst_caps_unref (vcaps);
    } else {
      return NULL;
    }
  } else {
    NVGSTDS_ERR_MSG_V ("Failed to get caps from sdp message");
    return NULL;
  }
  gst_caps_unref (caps);

  if (sinkConfig->type == NMOS_UDP_SINK_NV) {
    const gchar *localIfaceIp = gst_sdp_media_get_attribute_val (media, "x-nvds-iface-ip");
    if (!localIfaceIp) {
      NVGSTDS_ERR_MSG_V ("No attribute x-nvds-iface-ip in sdp message");
      return NULL;
    }
    if (!sinkConfig->packetsPerLine) {
      NVGSTDS_ERR_MSG_V ("wrong or missing value for 'packets-per-line' field in config file");
      return NULL;
    }
    if (!sinkConfig->payloadSize) {
      NVGSTDS_ERR_MSG_V ("wrong or missing value for 'payload-size' field in config file");
      return NULL;
    }
    g_object_set (G_OBJECT (sink), "local-iface-ip", localIfaceIp, "sdp-file",
        sinkConfig->sdpFile, "packets-per-line", sinkConfig->packetsPerLine,
        "payload-size", sinkConfig->payloadSize, NULL);

    g_object_set (G_OBJECT (payloader), "mtu", sinkConfig->payloadSize, NULL);
  }
  g_object_set (G_OBJECT(source), "pattern", 18, NULL);
  g_object_set (G_OBJECT(vparse), "use-sink-caps", 1, NULL);
  g_object_set (G_OBJECT(sink), "host", connection->address, "port", media->port, NULL);

  gst_bin_add_many (GST_BIN(videobin), source, capsfilter, queue, payloader, sink, NULL);

  if (!gst_element_link_many (source, capsfilter, queue, payloader, sink, NULL)) {
    NVGSTDS_ERR_MSG_V ("Error in linking components of video sender pipeline.");
    return NULL;
  }

  NvDsNmosSinkBin *sinkBin = g_new0 (NvDsNmosSinkBin, 1);
  sinkBin->bin = videobin;
  sinkBin->queue = queue;
  sinkBin->payloader = payloader;
  sinkBin->sink = sink;
  sinkBin->mediaType = g_strdup ("video");
  sinkBin->id = g_strdup (sinkConfig->id);

  return sinkBin;
}

static gboolean
get_media_conn_from_sdp_txt (gchar *sdpTxt, gchar *mediaType, GstSDPMessage **sdpMsg,
 const GstSDPMedia **media, const GstSDPConnection **conn)
{
  GstSDPResult result;
  GstSDPMessage *srcSdpMsg = NULL;

  result = gst_sdp_message_new_from_text (sdpTxt, &srcSdpMsg);
  if (result != GST_SDP_OK) {
    NVGSTDS_ERR_MSG_V ("Error (%d) in creating sdp message.", result);
    return FALSE;
  }

  const GstSDPMedia *srcMedia = get_sdp_media (srcSdpMsg, mediaType);
  if (!srcMedia) {
    NVGSTDS_ERR_MSG_V ("Error!! No media in sdp message");
    gst_sdp_message_free (srcSdpMsg);
    return FALSE;
  }

  const GstSDPConnection *srcConn = gst_sdp_media_get_connection (srcMedia, 0);
  if (!srcConn) {
    gst_sdp_message_free (srcSdpMsg);
    NVGSTDS_ERR_MSG_V ("Error!! No connection info in sdp message");
    return FALSE;
  }

  *sdpMsg = srcSdpMsg;
  *media = srcMedia;
  *conn = srcConn;

  return TRUE;
}

static gpointer
create_audio_recv_send_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSrcConfig *srcConfig)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (srcConfig != NULL, FALSE);

  GstElement *source = NULL;
  GstElement *sink = NULL;
  gboolean result;
  GstSDPMessage *srcSdpMsg = NULL;
  GstSDPMessage *sinkSdpMsg = NULL;
  const GstSDPMedia *srcMedia = NULL;
  const GstSDPConnection *srcConn = NULL;
  const GstSDPMedia *sinkMedia = NULL;
  const GstSDPConnection *sinkConn = NULL;

  result = get_media_conn_from_sdp_txt (srcConfig->srcSdpTxt, "audio",
              &srcSdpMsg, &srcMedia, &srcConn);
  if (!result) {
    NVGSTDS_ERR_MSG_V ("Error in parsing sdp txt.");
    return NULL;
  }

  result = get_media_conn_from_sdp_txt (srcConfig->sinkSdpTxt, "audio",
              &sinkSdpMsg, &sinkMedia, &sinkConn);
  if (!result) {
    NVGSTDS_ERR_MSG_V ("Error in parsing sdp txt.");
    gst_sdp_message_free (srcSdpMsg);
    return NULL;
  }

  GstElement *audiobin = gst_bin_new ("nvds-audio-bin");
  if (srcConfig->type == NMOS_UDP_SRC_NV) {
    source = gst_element_factory_make ("nvdsudpsrc", NULL);
  } else {
    source = gst_element_factory_make ("udpsrc", NULL);
  }

  GstElement *depay = gst_element_factory_make ("rtpL24depay", NULL);
  GstElement *aparse = gst_element_factory_make ("rawaudioparse", NULL);
  GstElement *queue = gst_element_factory_make ("queue", NULL);
  GstElement *capsfilter = gst_element_factory_make ("capsfilter", NULL);
  GstElement *payloader = gst_element_factory_make ("rtpL24pay", NULL);

  if (srcConfig->sinkType == NMOS_UDP_SINK_NV)
    sink = gst_element_factory_make ("nvdsudpsink", NULL);
  else
    sink = gst_element_factory_make ("udpsink", NULL);

  if (!source || !depay || !aparse || !capsfilter || !payloader || !sink || !queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create one of the component of audio pipeline.\n");
    goto error;
  }

  gint pt = atoi (gst_sdp_media_get_format (srcMedia, 0));
  GstCaps *caps = gst_sdp_media_get_caps_from_media (srcMedia, pt);
  if (caps) {
    // Replace application/x-unknown to application/x-rtp
    GstStructure *structure = gst_caps_get_structure (caps, 0);
    gst_structure_set_name (structure, "application/x-rtp");
  } else {
    NVGSTDS_ERR_MSG_V ("Failed to get caps from sdp message");
    goto error;
  }
  g_object_set (G_OBJECT(source), "caps", caps, NULL);
  gst_caps_unref (caps);

  pt = atoi (gst_sdp_media_get_format (sinkMedia, 0));
  caps = gst_sdp_media_get_caps_from_media (sinkMedia, pt);
  if (caps) {
    GstCaps *acaps = get_audio_caps_from_sdp_caps (caps);
    if (acaps) {
      g_object_set (G_OBJECT(capsfilter), "caps", acaps, NULL);
      gst_caps_unref (acaps);
    } else {
      gst_caps_unref (caps);
      goto error;
    }
  } else {
    NVGSTDS_ERR_MSG_V ("Failed to get caps from sdp message");
    goto error;
  }
  gst_caps_unref (caps);

  const gchar *localIfaceIp = gst_sdp_media_get_attribute_val (srcMedia, "x-nvds-iface-ip");
  if (!localIfaceIp) {
    NVGSTDS_ERR_MSG_V ("No attribute x-nvds-iface-ip in sdp message");
    goto error;
  }

  if (srcConfig->type == NMOS_UDP_SRC_NV) {
    g_object_set (G_OBJECT(source), "local-iface-ip", localIfaceIp, NULL);

    const gchar *srcFilter = gst_sdp_media_get_attribute_val (srcMedia, "source-filter");
    if (srcFilter) {
      gchar *tmpFilter = g_strdup (srcFilter);
      gchar **tokens = g_strsplit (g_strstrip (tmpFilter), " ", 0);
      gchar **tmp = tokens;
      guint count = 0;
      while (*tmp++) {
        count++;
      }
      if (count == 5) {
        g_object_set (G_OBJECT(source), "source-address", tokens[4], NULL);
      } else {
        NVGSTDS_WARN_MSG_V ("wrong value for attribute source-filter %s", srcFilter);
      }
      g_free (tmpFilter);
      g_strfreev (tokens);
    }
  }

  if (srcConfig->sinkType == NMOS_UDP_SINK_NV) {
    g_object_set (G_OBJECT(sink), "local-iface-ip", localIfaceIp, "sdp-file",
        srcConfig->sinkSdpFile, NULL);

    const gchar *ptime = gst_sdp_media_get_attribute_val (sinkMedia, "ptime");
    if (!ptime) {
      GST_ERROR ("No attribute ptime in sdp message");
      goto error;
    }

    gdouble tmpPtime = atof (ptime);
    if (tmpPtime <= 0) {
      GST_ERROR ("wrong value of ptime - %s", ptime);
      goto error;
    }
    guint64 ptime_ns = (guint64) (tmpPtime * 1000000 + 0.5);
    g_object_set (G_OBJECT (payloader), "max-ptime", ptime_ns, "ptime-multiple",
        ptime_ns, NULL);
  }

  g_object_set (G_OBJECT(source), "address", srcConn->address, "port",
      srcMedia->port, NULL);
  g_object_set (G_OBJECT(sink), "host", sinkConn->address, "port",
      sinkMedia->port, NULL);

  gst_sdp_message_free (srcSdpMsg);
  gst_sdp_message_free (sinkSdpMsg);

  g_object_set (G_OBJECT(aparse), "use-sink-caps", 1, NULL);

  gst_bin_add_many (GST_BIN(audiobin), source, depay, aparse,
        queue, capsfilter, payloader, sink, NULL);

  if (!gst_element_link_many (source, depay, aparse, capsfilter, queue,
      payloader, sink, NULL)) {
    NVGSTDS_ERR_MSG_V ("Error in linking components\n");
    return NULL;
  }

  NvDsNmosSrcBin *srcBin = g_new0 (NvDsNmosSrcBin, 1);
  srcBin->bin = audiobin;
  srcBin->src = source;
  srcBin->sink = sink;
  srcBin->queue = queue;
  srcBin->payloader = payloader;
  srcBin->mediaType = g_strdup ("audio");
  srcBin->srcId = g_strdup (srcConfig->id);

  return srcBin;

error:
  if (srcSdpMsg)
    gst_sdp_message_free (srcSdpMsg);
  if (sinkSdpMsg)
    gst_sdp_message_free (sinkSdpMsg);

  return NULL;
}

static gpointer
create_video_recv_send_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSrcConfig *srcConfig)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (srcConfig != NULL, FALSE);

  GstElement *source = NULL;
  GstElement *sink = NULL;
  GstElement *pgie = NULL;
  GstElement *nvosd = NULL;

  gboolean result;
  GstSDPMessage *srcSdpMsg = NULL;
  GstSDPMessage *sinkSdpMsg = NULL;
  const GstSDPMedia *srcMedia = NULL;
  const GstSDPConnection *srcConn = NULL;
  const GstSDPMedia *sinkMedia = NULL;
  const GstSDPConnection *sinkConn = NULL;

  result = get_media_conn_from_sdp_txt (srcConfig->srcSdpTxt, "video",
              &srcSdpMsg, &srcMedia, &srcConn);
  if (!result) {
    NVGSTDS_ERR_MSG_V ("Error in parsing sdp txt.");
    return NULL;
  }

  result = get_media_conn_from_sdp_txt (srcConfig->sinkSdpTxt, "video",
              &sinkSdpMsg, &sinkMedia, &sinkConn);
  if (!result) {
    NVGSTDS_ERR_MSG_V ("Error in parsing sdp txt.");
    gst_sdp_message_free (srcSdpMsg);
    return NULL;
  }

  GstElement *videobin = gst_bin_new ("nvds-video-bin");

  if (srcConfig->type == NMOS_UDP_SRC_NV) {
    source = gst_element_factory_make ("nvdsudpsrc", "video-udp-source");
  } else {
    source = gst_element_factory_make ("udpsrc", "video-udp-source");
  }

  GstElement *depay = gst_element_factory_make ("rtpvrawdepay", NULL);
  GstElement *payloader = gst_element_factory_make ("rtpvrawpay", NULL);
  GstElement *vparse = gst_element_factory_make ("rawvideoparse", NULL);
  GstElement *streammux = gst_element_factory_make ("nvstreammux", NULL);
  GstElement *nvvidconv = gst_element_factory_make ("nvvideoconvert", NULL);
  GstElement *nvvidconv2 = gst_element_factory_make ("nvvideoconvert", NULL);
  GstElement *capsfilter = gst_element_factory_make ("capsfilter", NULL);
  GstElement *queue = gst_element_factory_make ("queue", NULL);

  if (appCtx->config.enablePgie) {
    pgie = gst_element_factory_make ("nvinfer", NULL);
    nvosd = gst_element_factory_make ("nvdsosd", NULL);
  }

  GstElement *queue2 = gst_element_factory_make ("queue", NULL);
  if (srcConfig->sinkType == NMOS_UDP_SINK_NV)
    sink = gst_element_factory_make ("nvdsudpsink", NULL);
  else
    sink = gst_element_factory_make ("udpsink", NULL);

  if (!source || !depay || !vparse || !capsfilter || !payloader || !sink || !queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create one of the component of video pipeline.");
    goto error;
  }

  gint pt = atoi (gst_sdp_media_get_format (srcMedia, 0));
  GstCaps *caps = gst_sdp_media_get_caps_from_media (srcMedia, pt);
  guint width, height;
  if (caps) {
    const gchar *str;
    GstStructure *structure = gst_caps_get_structure (caps, 0);
    if (!(str = gst_structure_get_string (structure, "width"))) {
      NVGSTDS_ERR_MSG_V ("No width in sdp message");
      gst_caps_unref (caps);
      goto error;
    }
    width = atoi (str);

    if (!(str = gst_structure_get_string(structure, "height"))) {
      NVGSTDS_ERR_MSG_V("No height in sdp message");
      gst_caps_unref (caps);
      goto error;
    }
    height = atoi(str);
    // Replace application/x-unknown to application/x-rtp
    gst_structure_set_name (structure, "application/x-rtp");
  } else {
    NVGSTDS_ERR_MSG_V ("Failed to get caps from sdp message");
    goto error;
  }
  g_object_set (G_OBJECT(source), "caps", caps, NULL);
  gst_caps_unref (caps);

  pt = atoi (gst_sdp_media_get_format (sinkMedia, 0));
  caps = gst_sdp_media_get_caps_from_media (sinkMedia, pt);
  if (caps) {
    GstCaps *vcaps = get_video_caps_from_sdp_caps (caps);
    if (vcaps) {
      g_object_set (G_OBJECT(capsfilter), "caps", vcaps, NULL);
      gst_caps_unref (vcaps);
    } else {
      gst_caps_unref (caps);
      goto error;
    }
  } else {
    NVGSTDS_ERR_MSG_V ("Failed to get caps from sdp message");
    goto error;
  }
  gst_caps_unref (caps);

  const gchar *localIfaceIp = gst_sdp_media_get_attribute_val (srcMedia, "x-nvds-iface-ip");
  if (!localIfaceIp) {
    NVGSTDS_ERR_MSG_V ("No attribute x-nvds-iface-ip in sdp message");
    goto error;
  }

  if (srcConfig->type == NMOS_UDP_SRC_NV) {
    g_object_set (G_OBJECT(source), "local-iface-ip", localIfaceIp, NULL);

    const gchar *srcFilter = gst_sdp_media_get_attribute_val (srcMedia, "source-filter");
    if (srcFilter) {
      gchar *tmpFilter = g_strdup (srcFilter);
      gchar **tokens = g_strsplit (g_strstrip (tmpFilter), " ", 0);
      gchar **tmp = tokens;
      guint count = 0;
      while (*tmp++) {
        count++;
      }
      if (count == 5) {
        g_object_set (G_OBJECT(source), "source-address", tokens[4], NULL);
      } else {
        NVGSTDS_WARN_MSG_V ("wrong value for attribute source-filter %s", srcFilter);
      }
      g_free (tmpFilter);
      g_strfreev (tokens);
    }
  }

  if (srcConfig->sinkType == NMOS_UDP_SINK_NV) {
    if (!srcConfig->packetsPerLine) {
      NVGSTDS_ERR_MSG_V ("wrong or missing value for 'packets-per-line' field in config file");
      goto error;
    }
    if (!srcConfig->payloadSize) {
      NVGSTDS_ERR_MSG_V ("wrong or missing value for 'payload-size' field in config file");
      goto error;
    }
    g_object_set (G_OBJECT (sink), "local-iface-ip", localIfaceIp, "sdp-file",
        srcConfig->sinkSdpFile, "packets-per-line", srcConfig->packetsPerLine,
        "payload-size", srcConfig->payloadSize, NULL);

    g_object_set (G_OBJECT (payloader), "mtu", srcConfig->payloadSize, NULL);
  }

  g_object_set (G_OBJECT(source), "address", srcConn->address, "port",
        srcMedia->port, NULL);
  g_object_set (G_OBJECT(sink), "host", sinkConn->address, "port",
        sinkMedia->port, NULL);

  gst_sdp_message_free (srcSdpMsg);
  gst_sdp_message_free (sinkSdpMsg);

  g_object_set (G_OBJECT(vparse), "use-sink-caps", 1, NULL);

  g_object_set (G_OBJECT (streammux), "batch-size", 1, "nvbuf-memory-type", 0, NULL);
  g_object_set (G_OBJECT (streammux), "width", width, "height", height, NULL);
  if (appCtx->config.enablePgie) {
    g_object_set (G_OBJECT (pgie),
      "config-file-path", appCtx->config.pgieConfFile, NULL);

    gst_bin_add_many (GST_BIN(videobin), source, depay, vparse,
        nvvidconv, streammux, pgie, nvosd, queue, nvvidconv2, capsfilter,
        queue2, payloader, sink, NULL);
  } else {
    gst_bin_add_many (GST_BIN(videobin), source, depay, vparse,
        nvvidconv, streammux, queue, nvvidconv2, capsfilter, queue2, payloader,
        sink, NULL);
  }
  if (!gst_element_link_many (source, depay, vparse, nvvidconv, NULL)) {
    NVGSTDS_ERR_MSG_V ("Error in linking components of video pipeline.");
    return NULL;
  }
  if (appCtx->config.enablePgie) {
    if (!gst_element_link_many (streammux, pgie, nvosd, queue, nvvidconv2,
        capsfilter, queue2, payloader, sink, NULL)) {
      NVGSTDS_ERR_MSG_V("Error in linking components of video pipeline");
      return NULL;
    }
  } else {
    if (!gst_element_link_many (streammux, queue, nvvidconv2, capsfilter,
        queue2, payloader, sink, NULL)) {
      NVGSTDS_ERR_MSG_V("Error in linking components of video pipeline");
      return NULL;
    }
  }
  GstPad *sinkpad, *srcpad;
  sinkpad = gst_element_get_request_pad (streammux, "sink_0");
  if (!sinkpad) {
    NVGSTDS_ERR_MSG_V ("Streammux request sink pad failed.");
    return NULL;
  }

  srcpad = gst_element_get_static_pad (nvvidconv, "src");
  if (!srcpad) {
    NVGSTDS_ERR_MSG_V ("Decoder request src pad failed.");
    return NULL;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link decoder to stream muxer.");
    return NULL;
  }
  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  NvDsNmosSrcBin *srcBin = g_new0 (NvDsNmosSrcBin, 1);
  srcBin->bin = videobin;
  srcBin->src = source;
  srcBin->sink = sink;
  srcBin->queue = queue2;
  srcBin->payloader = payloader;
  srcBin->mediaType = g_strdup ("video");
  srcBin->srcId = g_strdup (srcConfig->id);

  return srcBin;

error:
  if (srcSdpMsg)
    gst_sdp_message_free (srcSdpMsg);
  if (sinkSdpMsg)
    gst_sdp_message_free (sinkSdpMsg);

  return NULL;
}

static gpointer
create_audio_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSrcConfig *srcConfig,
      const GstSDPMedia *media)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (srcConfig != NULL, FALSE);
  g_return_val_if_fail (media != NULL, FALSE);

  GstElement *source = NULL;

  const GstSDPConnection *connection = gst_sdp_media_get_connection (media, 0);
  if (!connection) {
    NVGSTDS_ERR_MSG_V ("Error!! No connection info in sdp message");
    return NULL;
  }

  GstElement *audiobin = gst_bin_new ("nvds-audio-bin");
  if (srcConfig->type == NMOS_UDP_SRC_NV) {
    source = gst_element_factory_make ("nvdsudpsrc", "audio-udp-src");
  } else {
    source = gst_element_factory_make ("udpsrc", "audio-udp-src");
  }

  GstElement *depay = gst_element_factory_make ("rtpL24depay", "araw-depay");
  GstElement *aparse = gst_element_factory_make ("rawaudioparse", "araw-parser");
  GstElement *queue = gst_element_factory_make ("queue", "audio-queue");
  GstElement *sink = gst_element_factory_make ("autoaudiosink", "audio-sink");

  if (!source || !depay || !aparse || !sink || !queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create one of the component of audio pipeline.\n");
    return NULL;
  }

  g_object_set (G_OBJECT(source), "address", connection->address, "port", media->port, NULL);

  gint pt = atoi (gst_sdp_media_get_format (media, 0));
  GstCaps *caps = gst_sdp_media_get_caps_from_media (media, pt);
  if (caps) {
    // Replace application/x-unknown to application/x-rtp
    GstStructure *structure = gst_caps_get_structure (caps, 0);
    gst_structure_set_name (structure, "application/x-rtp");
  } else {
    NVGSTDS_ERR_MSG_V ("Failed to get caps from sdp message");
    return NULL;
  }

  NVGSTDS_INFO_MSG_V ("audio caps: %s", gst_caps_to_string (caps));
  if (srcConfig->type == NMOS_UDP_SRC_NV) {
    const gchar *localIfaceIp = gst_sdp_media_get_attribute_val (media, "x-nvds-iface-ip");
    if (!localIfaceIp) {
      NVGSTDS_ERR_MSG_V ("No attribute x-nvds-iface-ip in sdp message");
      return NULL;
    }
    g_object_set (G_OBJECT(source), "local-iface-ip", localIfaceIp, NULL);

    const gchar *srcFilter = gst_sdp_media_get_attribute_val (media, "source-filter");
    if (srcFilter) {
      gchar *tmpFilter = g_strdup (srcFilter);
      gchar **tokens = g_strsplit (g_strstrip (tmpFilter), " ", 0);
      gchar **tmp = tokens;
      guint count = 0;
      while (*tmp++) {
        count++;
      }
      if (count == 5) {
        g_object_set (G_OBJECT(source), "source-address", tokens[4], NULL);
      } else {
        NVGSTDS_WARN_MSG_V ("wrong value for attribute source-filter %s", srcFilter);
      }
      g_free (tmpFilter);
      g_strfreev (tokens);
    }
  }
  g_object_set (G_OBJECT(source), "caps", caps, NULL);
  gst_caps_unref (caps);
  g_object_set (G_OBJECT(aparse), "use-sink-caps", 1, NULL);
  g_object_set (G_OBJECT(sink), "sync", 0, NULL);

  gst_bin_add_many (GST_BIN(audiobin), source, depay, aparse,
        queue, sink, NULL);

  if (!gst_element_link_many (source, depay, aparse, queue, sink, NULL)) {
    NVGSTDS_ERR_MSG_V ("Error in linking components\n");
    return NULL;
  }

  NvDsNmosSrcBin *srcBin = g_new0 (NvDsNmosSrcBin, 1);
  srcBin->bin = audiobin;
  srcBin->src = source;
  srcBin->mediaType = g_strdup ("audio");
  srcBin->srcId = g_strdup (srcConfig->id);

  return srcBin;
}

static gpointer
create_video_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSrcConfig *srcConfig,
      const GstSDPMedia *media)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (srcConfig != NULL, FALSE);
  g_return_val_if_fail (media != NULL, FALSE);

  GstElement *source = NULL;
  GstElement *pgie = NULL;
  GstElement *nvosd = NULL;

  const GstSDPConnection *connection = gst_sdp_media_get_connection (media, 0);
  if (!connection) {
    NVGSTDS_ERR_MSG_V ("Error!! No connection info in sdp message");
    return NULL;
  }

  GstElement *videobin = gst_bin_new ("nvds-video-bin");

  if (srcConfig->type == NMOS_UDP_SRC_NV) {
    source = gst_element_factory_make ("nvdsudpsrc", "video-udp-source");
  } else {
    source = gst_element_factory_make ("udpsrc", "video-udp-source");
  }

  GstElement *depay = gst_element_factory_make ("rtpvrawdepay", "vraw-depay");
  GstElement *vparse = gst_element_factory_make ("rawvideoparse", "vraw-parser");
  GstElement *streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  GstElement *nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  if (appCtx->config.enablePgie) {
    pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  }
  GstElement *queue = gst_element_factory_make ("queue", "que1");
  GstElement *sink = gst_element_factory_make ("nveglglessink", "vsink");

  if (!source || !depay || !vparse || !sink || !queue) {
    NVGSTDS_ERR_MSG_V ("Failed to create one of the component of video pipeline.");
    return NULL;
  }

  g_object_set (G_OBJECT(source), "address", connection->address, "port", media->port, NULL);
  gint pt = atoi (gst_sdp_media_get_format (media, 0));
  GstCaps *caps = gst_sdp_media_get_caps_from_media (media, pt);
  guint width, height;
  if (caps) {
    const gchar *str;
    GstStructure *structure = gst_caps_get_structure (caps, 0);
    if (!(str = gst_structure_get_string (structure, "width"))) {
      NVGSTDS_ERR_MSG_V ("No width in sdp message");
      return NULL;
    }
    width = atoi (str);

    if (!(str = gst_structure_get_string(structure, "height"))) {
      NVGSTDS_ERR_MSG_V("No height in sdp message");
      return NULL;
    }
    height = atoi(str);
    // Replace application/x-unknown to application/x-rtp
    gst_structure_set_name (structure, "application/x-rtp");
  } else {
    NVGSTDS_ERR_MSG_V ("Failed to get caps from sdp message");
    return NULL;
  }
  NVGSTDS_INFO_MSG_V ("received caps: %s\n", gst_caps_to_string (caps));
  if (srcConfig->type == NMOS_UDP_SRC_NV) {
    const gchar *localIfaceIp = gst_sdp_media_get_attribute_val (media, "x-nvds-iface-ip");
    if (!localIfaceIp) {
      NVGSTDS_ERR_MSG_V ("No attribute x-nvds-iface-ip in sdp message");
      return NULL;
    }
    g_object_set (G_OBJECT(source), "local-iface-ip", localIfaceIp, NULL);

    const gchar *srcFilter = gst_sdp_media_get_attribute_val (media, "source-filter");
    if (srcFilter) {
      gchar *tmpFilter = g_strdup (srcFilter);
      gchar **tokens = g_strsplit (g_strstrip (tmpFilter), " ", 0);
      gchar **tmp = tokens;
      guint count = 0;
      while (*tmp++) {
        count++;
      }
      if (count == 5) {
        g_object_set (G_OBJECT (source), "source-address", tokens[4], NULL);
      } else {
        NVGSTDS_WARN_MSG_V ("wrong value for attribute source-filter %s", srcFilter);
      }
      g_free (tmpFilter);
      g_strfreev (tokens);
    }
  }

  g_object_set (G_OBJECT(source), "caps", caps, NULL);
  gst_caps_unref (caps);

  g_object_set (G_OBJECT(vparse), "use-sink-caps", 1, NULL);
  g_object_set (G_OBJECT(sink), "sync", 0, NULL);

  g_object_set (G_OBJECT (streammux), "batch-size", 1, "nvbuf-memory-type", 0, NULL);
  g_object_set (G_OBJECT (streammux), "width", width, "height", height, NULL);

  if (appCtx->config.enablePgie) {
    g_object_set (G_OBJECT (pgie),
      "config-file-path", appCtx->config.pgieConfFile, NULL);

    gst_bin_add_many (GST_BIN(videobin), source, depay, vparse,
        nvvidconv, streammux, pgie, nvosd, queue, sink, NULL);
  } else {
    gst_bin_add_many (GST_BIN(videobin), source, depay, vparse,
        nvvidconv, streammux, queue, sink, NULL);
  }

  if (!gst_element_link_many (source, depay, vparse, nvvidconv, NULL)) {
    NVGSTDS_ERR_MSG_V ("Error in linking components of video pipeline.");
    return NULL;
  }

  if (appCtx->config.enablePgie) {
    if (!gst_element_link_many (streammux, pgie, nvosd, queue, sink, NULL)) {
      NVGSTDS_ERR_MSG_V("Error in linking components of video pipeline");
      return NULL;
    }
  } else {
    if (!gst_element_link_many (streammux, queue, sink, NULL)) {
      NVGSTDS_ERR_MSG_V("Error in linking components of video pipeline");
      return NULL;
    }
  }

  GstPad *sinkpad, *srcpad;
  sinkpad = gst_element_get_request_pad (streammux, "sink_0");
  if (!sinkpad) {
    NVGSTDS_ERR_MSG_V ("Streammux request sink pad failed.");
    return NULL;
  }

  srcpad = gst_element_get_static_pad (nvvidconv, "src");
  if (!srcpad) {
    NVGSTDS_ERR_MSG_V ("Decoder request src pad failed.");
    return NULL;
  }

  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
    NVGSTDS_ERR_MSG_V ("Failed to link decoder to stream muxer.");
    return NULL;
  }

  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  NvDsNmosSrcBin *srcBin = g_new0 (NvDsNmosSrcBin, 1);
  srcBin->bin = videobin;
  srcBin->src = source;
  srcBin->mediaType = g_strdup ("video");
  srcBin->srcId = g_strdup (srcConfig->id);

  return srcBin;
}

static gboolean
update_sender_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSinkConfig *sinkConfig,
      const gchar *sdpTxt)
{
  g_return_val_if_fail (sinkConfig, FALSE);
  g_return_val_if_fail (sdpTxt, FALSE);
  g_return_val_if_fail (appCtx, FALSE);

  if (g_hash_table_size (appCtx->sinks)) {
    NvDsNmosSinkBin *sinkBin;
    sinkBin = (NvDsNmosSinkBin *) g_hash_table_lookup (appCtx->sinks, sinkConfig->id);
    if (sinkBin) {
      EventHandleData *evData = g_new0 (EventHandleData, 1);
      evData->queue = sinkBin->queue;
      evData->nextElem = sinkBin->payloader;
      evData->sink = sinkBin->sink;
      return update_active_component (sinkBin->sink, sdpTxt, TRUE, evData);
    }
  }

  GstSDPResult result;
  GstSDPMessage *sdpMsg;
  NvDsNmosSinkBin *sinkBin = NULL;
  result = gst_sdp_message_new_from_text (sdpTxt, &sdpMsg);
  if (result != GST_SDP_OK) {
    NVGSTDS_ERR_MSG_V ("Error (%d) in creating sdp message.", result);
    return FALSE;
  }

  const GstSDPMedia *media = gst_sdp_message_get_media (sdpMsg, 0);
  if (!media) {
    NVGSTDS_ERR_MSG_V ("Error!! No media in sdp message");
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  if (!g_strcmp0 (media->media, "video")) {
    sinkBin = (NvDsNmosSinkBin *) create_video_sender_pipeline (appCtx, sinkConfig, media);
    if (!sinkBin) {
      NVGSTDS_ERR_MSG_V ("Error in creating video sender bin");
      gst_sdp_message_free (sdpMsg);
      return FALSE;
    }
  } else if (!g_strcmp0 (media->media, "audio")) {
    sinkBin = (NvDsNmosSinkBin *) create_audio_sender_pipeline (appCtx, sinkConfig, media);
    if (!sinkBin) {
      NVGSTDS_ERR_MSG_V ("Error in creating audio sender bin");
      gst_sdp_message_free (sdpMsg);
      return FALSE;
    }
  } else {
    NVGSTDS_INFO_MSG_V ("media %s not supported", media->media);
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  gst_sdp_message_free (sdpMsg);
  if (!appCtx->isPipelineActive) {
    appCtx->pipeline = gst_pipeline_new ("nvds-nmos-pipeline");
    GstBus *bus = gst_pipeline_get_bus (GST_PIPELINE(appCtx->pipeline));
    appCtx->watchId = gst_bus_add_watch (bus, bus_call, appCtx);
    gst_object_unref (bus);
  }

  g_hash_table_insert (appCtx->sinks, g_strdup (sinkConfig->id), sinkBin);
  gst_bin_add (GST_BIN(appCtx->pipeline), sinkBin->bin);
  if (!appCtx->isPipelineActive) {
    gst_element_set_state (GST_ELEMENT(appCtx->pipeline), GST_STATE_PLAYING);
    appCtx->isPipelineActive = TRUE;
  } else {
    gst_element_sync_state_with_parent (sinkBin->bin);
  }

  return TRUE;
}

static gboolean
remove_sender_pipeline (NvDsNmosAppCtx *appCtx, const gchar *sinkId)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (sinkId != NULL, FALSE);

  NvDsNmosSinkBin *sinkBin;

  sinkBin = (NvDsNmosSinkBin *) g_hash_table_lookup (appCtx->sinks, sinkId);
  if (!sinkBin) {
    NVGSTDS_INFO_MSG_V ("Receiver %s is not active", sinkId);
    return TRUE;
  }

  NVGSTDS_INFO_MSG_V ("removing %s %s bin from the pipeline\n", sinkId, sinkBin->mediaType);
  if (gst_element_set_state (sinkBin->bin,
                              GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (sinkBin->bin, "Can't set bin to NULL");
    return FALSE;
  }

  if (!gst_bin_remove (GST_BIN(appCtx->pipeline), sinkBin->bin)) {
    g_print ("Can't remove the %s %s bin from pipeline\n", sinkId, sinkBin->mediaType);
    return FALSE;
  }

  g_free (sinkBin->id);
  g_free (sinkBin->mediaType);
  g_hash_table_remove (appCtx->sinks, sinkId);

  if (!g_hash_table_size (appCtx->sinks)) {
    gst_element_set_state (GST_ELEMENT(appCtx->pipeline), GST_STATE_NULL);
    gst_object_unref (GST_OBJECT (appCtx->pipeline));
    g_source_remove (appCtx->watchId);
    appCtx->pipeline = NULL;
    appCtx->isPipelineActive = FALSE;
  }
  return TRUE;
}

static gboolean
update_recv_send_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSrcConfig *srcConfig,
    const gchar *sdpTxt, gboolean isSink)
{
  g_return_val_if_fail (srcConfig, FALSE);
  g_return_val_if_fail (sdpTxt, FALSE);
  g_return_val_if_fail (appCtx, FALSE);

  if (g_hash_table_size (appCtx->sources)) {
    NvDsNmosSrcBin *srcBin;
    srcBin = (NvDsNmosSrcBin *) g_hash_table_lookup (appCtx->sources, srcConfig->id);
    if (srcBin) {
      if (isSink) {
        EventHandleData *evData = g_new0 (EventHandleData, 1);
        evData->queue = srcBin->queue;
        evData->nextElem = srcBin->payloader;
        evData->sink = srcBin->sink;
        return update_active_component (srcBin->sink, sdpTxt, isSink, evData);
      } else {
        return update_active_component (srcBin->src, sdpTxt, isSink, NULL);
      }
    }
  }

  if (!srcConfig->srcSdpTxt || !srcConfig->sinkSdpTxt) {
    // We still don't have config for both source and sink.
    // Return for now and create pipeline once both configs are available.
    return TRUE;
  }

  GstSDPResult result;
  GstSDPMessage *sdpMsg;
  NvDsNmosSrcBin *srcBin = NULL;
  result = gst_sdp_message_new_from_text (sdpTxt, &sdpMsg);
  if (result != GST_SDP_OK) {
    NVGSTDS_ERR_MSG_V ("Error (%d) in creating sdp message.", result);
    return FALSE;
  }

  const GstSDPMedia *media = gst_sdp_message_get_media (sdpMsg, 0);
  if (!media) {
    NVGSTDS_ERR_MSG_V ("Error!! No media in sdp message");
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  if (!g_strcmp0 (media->media, "video")) {
    srcBin = (NvDsNmosSrcBin *) create_video_recv_send_pipeline (appCtx,
                srcConfig);
    if (!srcBin) {
      NVGSTDS_ERR_MSG_V ("Error in creating video bin");
      gst_sdp_message_free (sdpMsg);
      return FALSE;
    }
  } else if (!g_strcmp0 (media->media, "audio")) {
    srcBin = (NvDsNmosSrcBin *) create_audio_recv_send_pipeline (appCtx,
                srcConfig);
    if (!srcBin) {
      NVGSTDS_ERR_MSG_V ("Error in creating audio bin");
      gst_sdp_message_free (sdpMsg);
      return FALSE;
    }
  } else {
    NVGSTDS_INFO_MSG_V ("media %s not supported", media->media);
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  gst_sdp_message_free (sdpMsg);
  if (!appCtx->isPipelineActive) {
    appCtx->pipeline = gst_pipeline_new ("nvds-nmos-pipeline");
    GstBus *bus = gst_pipeline_get_bus (GST_PIPELINE(appCtx->pipeline));
    appCtx->watchId = gst_bus_add_watch (bus, bus_call, appCtx);
    gst_object_unref (bus);
  }

  g_hash_table_insert (appCtx->sources, g_strdup (srcConfig->id), srcBin);
  gst_bin_add (GST_BIN(appCtx->pipeline), srcBin->bin);
  if (!appCtx->isPipelineActive) {
    gst_element_set_state (GST_ELEMENT(appCtx->pipeline), GST_STATE_PLAYING);
    appCtx->isPipelineActive = TRUE;
  } else {
    gst_element_sync_state_with_parent (srcBin->bin);
  }
  return TRUE;
}

static gboolean
remove_recv_send_pipeline (NvDsNmosAppCtx *appCtx, const gchar *srcId)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (srcId != NULL, FALSE);

  NvDsNmosSrcBin *srcBin;
  srcBin = (NvDsNmosSrcBin *) g_hash_table_lookup (appCtx->sources, srcId);
  if (!srcBin) {
    NVGSTDS_INFO_MSG_V ("%s is not active", srcId);
    return TRUE;
  }

  NVGSTDS_INFO_MSG_V ("removing %s bin from the pipeline\n", srcBin->mediaType);
  if (gst_element_set_state (srcBin->bin,
                              GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (srcBin->bin, "Can't set bin to NULL");
    return FALSE;
  }

  if (!gst_bin_remove (GST_BIN(appCtx->pipeline), srcBin->bin)) {
    NVGSTDS_INFO_MSG_V ("Can't remove the %s %s bin from pipeline\n", srcId,
        srcBin->mediaType);
    return FALSE;
  }

  g_free (srcBin->srcId);
  g_free (srcBin->mediaType);
  g_hash_table_remove (appCtx->sources, srcId);

  if (!g_hash_table_size (appCtx->sources)) {
    gst_element_set_state (GST_ELEMENT(appCtx->pipeline), GST_STATE_NULL);
    gst_object_unref (GST_OBJECT (appCtx->pipeline));
    g_source_remove (appCtx->watchId);
    appCtx->pipeline = NULL;
    appCtx->isPipelineActive = FALSE;
  }
  return TRUE;
}

static gboolean
update_receive_pipeline (NvDsNmosAppCtx *appCtx, NvDsNmosSrcConfig *srcConfig,
      const gchar *sdpTxt)
{
  g_return_val_if_fail (srcConfig, FALSE);
  g_return_val_if_fail (sdpTxt, FALSE);
  g_return_val_if_fail (appCtx, FALSE);

  if (g_hash_table_size (appCtx->sources)) {
    NvDsNmosSrcBin *srcBin;
    srcBin = (NvDsNmosSrcBin *) g_hash_table_lookup (appCtx->sources, srcConfig->id);
    if (srcBin) {
      // update the already active source
      return update_active_component (srcBin->src, sdpTxt, FALSE, NULL);
    }
  }

  GstSDPResult result;
  GstSDPMessage *sdpMsg;
  NvDsNmosSrcBin *srcBin = NULL;
  result = gst_sdp_message_new_from_text (sdpTxt, &sdpMsg);
  if (result != GST_SDP_OK) {
    NVGSTDS_ERR_MSG_V ("Error (%d) in creating sdp message.", result);
    return FALSE;
  }

  const GstSDPMedia *media = gst_sdp_message_get_media (sdpMsg, 0);
  if (!media) {
    NVGSTDS_ERR_MSG_V ("Error!! No media in sdp message");
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  if (!g_strcmp0 (media->media, "video")) {
    srcBin = (NvDsNmosSrcBin *) create_video_pipeline (appCtx, srcConfig, media);
    if (!srcBin) {
      NVGSTDS_ERR_MSG_V ("Error in creating video bin");
      gst_sdp_message_free (sdpMsg);
      return FALSE;
    }
  } else if (!g_strcmp0 (media->media, "audio")) {
    srcBin = (NvDsNmosSrcBin *) create_audio_pipeline (appCtx, srcConfig, media);
    if (!srcBin) {
      NVGSTDS_ERR_MSG_V ("Error in creating audio bin");
      gst_sdp_message_free (sdpMsg);
      return FALSE;
    }
  } else {
    NVGSTDS_INFO_MSG_V ("media %s not supported", media->media);
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  gst_sdp_message_free (sdpMsg);
  if (!appCtx->isPipelineActive) {
    appCtx->pipeline = gst_pipeline_new ("nvds-nmos-pipeline");
    GstBus *bus = gst_pipeline_get_bus (GST_PIPELINE(appCtx->pipeline));
    appCtx->watchId = gst_bus_add_watch (bus, bus_call, appCtx);
    gst_object_unref (bus);
  }

  g_hash_table_insert (appCtx->sources, g_strdup (srcConfig->id), srcBin);
  gst_bin_add (GST_BIN(appCtx->pipeline), srcBin->bin);
  if (!appCtx->isPipelineActive) {
    gst_element_set_state (GST_ELEMENT(appCtx->pipeline), GST_STATE_PLAYING);
    appCtx->isPipelineActive = TRUE;
  } else {
    gst_element_sync_state_with_parent (srcBin->bin);
  }
  return TRUE;
}

static gboolean
remove_receive_pipeline (NvDsNmosAppCtx *appCtx, const gchar *srcId)
{
  g_return_val_if_fail (appCtx != NULL, FALSE);
  g_return_val_if_fail (srcId != NULL, FALSE);

  NvDsNmosSrcBin *srcBin;

  srcBin = (NvDsNmosSrcBin *) g_hash_table_lookup (appCtx->sources, srcId);
  if (!srcBin) {
    NVGSTDS_INFO_MSG_V ("Receiver %s is not active", srcId);
    return TRUE;
  }

  NVGSTDS_INFO_MSG_V ("removing %s %s bin from the pipeline\n", srcId,
      srcBin->mediaType);
  if (gst_element_set_state (srcBin->bin,
                              GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE) {
    GST_ERROR_OBJECT (srcBin->bin, "Can't set bin to NULL");
    return FALSE;
  }

  if (!gst_bin_remove (GST_BIN(appCtx->pipeline), srcBin->bin)) {
    NVGSTDS_ERR_MSG_V ("Can't remove the %s %s bin from pipeline\n", srcId,
        srcBin->mediaType);
    return FALSE;
  }

  g_free (srcBin->srcId);
  g_free (srcBin->mediaType);
  g_hash_table_remove (appCtx->sources, srcId);

  if (!g_hash_table_size (appCtx->sources)) {
    gst_element_set_state (GST_ELEMENT(appCtx->pipeline), GST_STATE_NULL);
    gst_object_unref (GST_OBJECT (appCtx->pipeline));
    g_source_remove (appCtx->watchId);
    appCtx->pipeline = NULL;
    appCtx->isPipelineActive = FALSE;
  }
  return TRUE;
}

static void logFunc(
    NvDsNmosNodeServer *server,
    const char *categories,
    int level,
    const char *message)
{
    g_print ("%s [%d:%s]\n", message, level, categories);
}

static bool nmos_callback_handler (NvDsNmosNodeServer *server, const char *id,
      const char *sdp)
{
  if (!server) {
    NVGSTDS_ERR_MSG_V ("NULL nmos node server instance");
    return FALSE;
  }

  if (!id) {
    NVGSTDS_ERR_MSG_V ("NULL id for nmos sender / reciever");
    return FALSE;
  }

  NvDsNmosAppCtx *appCtx = (NvDsNmosAppCtx *) server->user_data;
  if (!appCtx) {
    NVGSTDS_ERR_MSG_V ("NULL user data");
    return FALSE;
  }

  if (g_AppMode == NVDS_NMOS_APP_MODE_RECVSEND) {
    guint i;
    NvDsNmosSrcConfig *srcConfig;
    gboolean isSink = FALSE;

    if (g_str_has_prefix (id, "sink")) {
      isSink = TRUE;
      for (i = 0; i < appCtx->config.numSrc; i++) {
        if (!g_strcmp0(id, appCtx->config.srcConfigs[i].sinkId))
          break;
      }

      if (i == appCtx->config.numSrc) {
        NVGSTDS_ERR_MSG_V("No sink with id %s\n", id);
        return FALSE;
      }
      srcConfig = &appCtx->config.srcConfigs[i];
      g_free (srcConfig->sinkSdpTxt);
      srcConfig->sinkSdpTxt = g_strdup (sdp);
    } else {
      for (i = 0; i < appCtx->config.numSrc; i++) {
        if (!g_strcmp0 (id, appCtx->config.srcConfigs[i].id))
          break;
      }

      if (i == appCtx->config.numSrc) {
        NVGSTDS_ERR_MSG_V("No source with id %s\n", id);
        return FALSE;
      }
      srcConfig = &appCtx->config.srcConfigs[i];
      g_free (srcConfig->srcSdpTxt);
      srcConfig->srcSdpTxt = g_strdup (sdp);
    }

    if (sdp) {
      return update_recv_send_pipeline (appCtx, srcConfig, sdp, isSink);
    } else {
      return remove_recv_send_pipeline (appCtx, srcConfig->id);
    }
  }

  if (g_str_has_prefix (id, "sink")) {
    // Handle event for sink
    if (sdp) {
      guint i;
      NvDsNmosSinkConfig *sinkConfig;

      for (i = 0; i < appCtx->config.numSink; i++) {
        if (!g_strcmp0(id, appCtx->config.sinkConfigs[i].id))
          break;
      }

      if (i == appCtx->config.numSink) {
        NVGSTDS_ERR_MSG_V("No sink with id %s\n", id);
        return FALSE;
      }
      sinkConfig = &appCtx->config.sinkConfigs[i];
      return update_sender_pipeline (appCtx, sinkConfig, sdp);
    } else {
      // Disable one of the running sink pipeline.
      return remove_sender_pipeline (appCtx, id);
    }
  }

  if (sdp) {
    // Enable / Update one of the source.
    guint i;
    NvDsNmosSrcConfig *srcConfig;

    for (i = 0; i < appCtx->config.numSrc; i++) {
      if (!g_strcmp0(id, appCtx->config.srcConfigs[i].id))
        break;
    }

    if (i == appCtx->config.numSrc) {
      NVGSTDS_ERR_MSG_V("No source with id %s\n", id);
      return FALSE;
    }
    srcConfig = &appCtx->config.srcConfigs[i];
    return update_receive_pipeline (appCtx, srcConfig, sdp);
  } else {
    // Disable one of the running source pipeline.
    return remove_receive_pipeline (appCtx, id);
  }
  return TRUE;
}

static gboolean
get_id_and_txt_from_file (gchar *sdpFile, gchar **txt, gchar **nvdsId)
{
  gchar *sdpTxt = NULL;
  GstSDPResult result;
  GstSDPMessage *sdpMsg;
  const gchar *id = NULL;

  if (!sdpFile) {
    NVGSTDS_ERR_MSG_V ("Missing sdp file");
    return FALSE;
  }

  if (!g_file_get_contents (sdpFile, &sdpTxt, NULL, NULL)) {
    NVGSTDS_ERR_MSG_V ("Error in reading contents of sdp file - %s\n", sdpFile);
    return FALSE;
  }

  result = gst_sdp_message_new_from_text (sdpTxt, &sdpMsg);
  if (result != GST_SDP_OK) {
    NVGSTDS_ERR_MSG_V ("Error (%d) in creating sdp message.", result);
    g_free (sdpTxt);
    return FALSE;
  }

  id = gst_sdp_message_get_attribute_val (sdpMsg, "x-nvds-id");
  if (!id) {
    NVGSTDS_ERR_MSG_V ("No x-nvds-id attribute found in sdp file %s", sdpFile);
    g_free (sdpTxt);
    gst_sdp_message_free (sdpMsg);
    return FALSE;
  }

  *nvdsId = g_strdup (id);
  *txt = sdpTxt;
  gst_sdp_message_free (sdpMsg);

  return TRUE;
}

int main (int argc, char *argv[])
{
  GOptionContext *optCtx = NULL;
  GOptionGroup *group = NULL;
  GError *error = NULL;
  guint i, count;
  gboolean ret;

  optCtx = g_option_context_new ("Nvidia DS-NMOS Demo");
  group = g_option_group_new (NULL, NULL, NULL, NULL, NULL);
  g_option_group_add_entries (group, entries);

  g_option_context_set_main_group (optCtx, group);
  g_option_context_add_group (optCtx, gst_init_get_option_group ());
  if (!g_option_context_parse (optCtx, &argc, &argv, &error)) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    g_error_free (error);
    g_option_context_free (optCtx);
    return -1;
  }
  g_option_context_free (optCtx);

  if (!cfgFile) {
    NVGSTDS_ERR_MSG_V ("specify config file with -c option");
    return -1;
  }

  NvDsNmosAppCtx *appCtx = g_new0 (NvDsNmosAppCtx, 1);
  appCtx->sources = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);
  appCtx->sinks = g_hash_table_new_full (g_str_hash, g_str_equal, g_free, g_free);

  if (!parse_config_file (appCtx, cfgFile)) {
    NVGSTDS_ERR_MSG_V ("Error in parsing config file");
    goto app_error;
  }

  NvDsNmosReceiverConfig *srcConfig;
  NvDsNmosSenderConfig *sinkConfig;
  NvDsNmosNodeConfig nodeConfig = { 0 };

  if (appCtx->config.hostName)
    nodeConfig.host_name = appCtx->config.hostName;
  else
    nodeConfig.host_name = "dsnode.local";

  if (appCtx->config.httpPort)
    nodeConfig.http_port = appCtx->config.httpPort;
  else
    nodeConfig.http_port = 8080;

  nodeConfig.seed = appCtx->config.seed;
  nodeConfig.rtp_connection_activated = &nmos_callback_handler;
  nodeConfig.log_callback = &logFunc;
  nodeConfig.log_level = 0;

  if (g_AppMode == NVDS_NMOS_APP_MODE_RECEIVE) {
    count = appCtx->config.numSrc;
    if (!count) {
      NVGSTDS_ERR_MSG_V ("No source mentioned in config file");
      goto app_error;
    }
    srcConfig = g_new0 (NvDsNmosReceiverConfig, count);
    nodeConfig.num_receivers = count;
    nodeConfig.receivers = srcConfig;

    for (i = 0; i < count; i++) {
      gchar *sdpTxt = NULL;
      gchar *nvdsId = NULL;

      ret = get_id_and_txt_from_file (appCtx->config.srcConfigs[i].sdpFile,
                &sdpTxt, &nvdsId);
      if (!ret)
        goto sdp_error;

      srcConfig[i].sdp = sdpTxt;
      appCtx->config.srcConfigs[i].id = nvdsId;
    }
  } else if (g_AppMode == NVDS_NMOS_APP_MODE_SEND) {
    count = appCtx->config.numSink;
    if (!count) {
      NVGSTDS_ERR_MSG_V ("No sink mentioned in config file");
      goto app_error;
    }
    sinkConfig = g_new0 (NvDsNmosSenderConfig, count);
    nodeConfig.num_senders = count;
    nodeConfig.senders = sinkConfig;

    for (i = 0; i < count; i++) {
      gchar *sdpTxt = NULL;
      gchar *nvdsId = NULL;

      ret = get_id_and_txt_from_file (appCtx->config.sinkConfigs[i].sdpFile,
                &sdpTxt, &nvdsId);
      if (!ret)
        goto sdp_error;

      sinkConfig[i].sdp = sdpTxt;
      appCtx->config.sinkConfigs[i].id = nvdsId;
    }
  } else if (g_AppMode == NVDS_NMOS_APP_MODE_RECVSEND) {
    count = appCtx->config.numSrc;
    if (!count) {
      NVGSTDS_ERR_MSG_V ("No source mentioned in config file");
      goto app_error;
    }
    srcConfig = g_new0 (NvDsNmosReceiverConfig, count);
    sinkConfig = g_new0 (NvDsNmosSenderConfig, count);
    nodeConfig.num_receivers = count;
    nodeConfig.receivers = srcConfig;
    nodeConfig.num_senders = count;
    nodeConfig.senders = sinkConfig;

    for (i = 0; i < count; i++) {
      gchar *sdpTxt = NULL;
      gchar *nvdsId = NULL;

      ret = get_id_and_txt_from_file (appCtx->config.srcConfigs[i].sdpFile,
                &sdpTxt, &nvdsId);
      if (!ret)
        goto sdp_error;

      srcConfig[i].sdp = sdpTxt;
      appCtx->config.srcConfigs[i].id = nvdsId;

      ret = get_id_and_txt_from_file (appCtx->config.srcConfigs[i].sinkSdpFile,
                &sdpTxt, &nvdsId);
      if (!ret)
        goto sdp_error;

      sinkConfig[i].sdp = sdpTxt;
      appCtx->config.srcConfigs[i].sinkId = nvdsId;
    }
  } else {
    NVGSTDS_ERR_MSG_V ("Unknown app mode - [%u]", g_AppMode);
    goto app_error;
  }

  NvDsNmosNodeServer nodeServer = { 0 };
  nodeServer.user_data = (gpointer) appCtx;

  gboolean status;
  status = create_nmos_node_server (&nodeConfig, &nodeServer);
  if (!status) {
    NVGSTDS_ERR_MSG_V ("Failed in creating nmos node server");
    goto sdp_error;
  }

  for (i = 0; i < nodeConfig.num_senders; i++) {
    g_free ((gpointer) nodeConfig.senders[i].sdp);
  }
  g_free (nodeConfig.senders);

  for (i = 0; i < nodeConfig.num_receivers; i++) {
    g_free ((gpointer) nodeConfig.receivers[i].sdp);
  }
  g_free (nodeConfig.receivers);

  appCtx->loop = g_main_loop_new (NULL, FALSE);

  _intr_setup ();
  g_timeout_add (400, check_for_interrupt, appCtx);

  NVGSTDS_INFO_MSG_V ("DS-NMOS Node started\n");

  g_timeout_add (40, event_thread_func, appCtx);
  g_main_loop_run (appCtx->loop);

  NVGSTDS_INFO_MSG_V ("stopping DS-NMOS node....\n");
  status = destroy_nmos_node_server (&nodeServer);
  if (!status) {
    NVGSTDS_ERR_MSG_V ("Failed to stop nmos node server");
  } else {
    NVGSTDS_INFO_MSG_V ("DS-NMOS node stopped....\n");
  }

  if (appCtx->isPipelineActive) {
    gst_element_set_state (GST_ELEMENT(appCtx->pipeline), GST_STATE_NULL);
    gst_object_unref (GST_OBJECT (appCtx->pipeline));
    g_source_remove (appCtx->watchId);
  }

  g_main_loop_unref (appCtx->loop);
  g_free (cfgFile);
  g_hash_table_unref (appCtx->sources);
  g_hash_table_unref (appCtx->sinks);
  g_free (appCtx);
  return 0;

sdp_error:
  for (i = 0; i < nodeConfig.num_senders; i++) {
    g_free ((gpointer) nodeConfig.senders[i].sdp);
  }
  g_free (nodeConfig.senders);

  for (i = 0; i < nodeConfig.num_receivers; i++) {
    g_free ((gpointer) nodeConfig.receivers[i].sdp);
  }
  g_free (nodeConfig.receivers);

app_error:
  g_free (cfgFile);
  g_hash_table_unref (appCtx->sources);
  g_hash_table_unref (appCtx->sinks);
  g_free (appCtx);
  return -1;
}
