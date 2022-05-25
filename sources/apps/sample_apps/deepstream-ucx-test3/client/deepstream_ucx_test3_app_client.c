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
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
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

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *ucxclientsrc = NULL, *caps_filter1 = NULL,
      *metaext = NULL, *streamdemux = NULL, *audioconv = NULL, *caps_filter2 =
      NULL, *waveenc = NULL, *filesink = NULL;

  GstCaps *caps1 = NULL, *caps2 = NULL;
  GstCapsFeatures *feature = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  gint64 port = 0;
  gchar *addr = NULL;
  gchar *audformat1 = "F32LE";
  gchar *audformat2 = "S16LE";
  gchar *endptr0 = NULL, *libpath = NULL, *outfile = NULL;

  int current_device = -1;
  cudaGetDevice (&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, current_device);

  /* Check input arguments */
  if (argc != 5) {
    g_printerr ("Usage: %s <addr> <port> "
        "<audio metadata serialization lib path>  <outputfile>\n", argv[0]);
    return -1;
  }

  addr = argv[1];
  port = g_ascii_strtoll (argv[2], &endptr0, 10);
  libpath = argv[3];
  outfile = argv[4];

  if (!g_hostname_is_ip_address (addr)) {
    g_printerr ("Addr specified is not a valid IP address/hostname: %s\n",
        addr);
    return -1;
  }

  if ((port == 0 && endptr0 == argv[2]) || port <= 0) {
    g_printerr ("Incorrect port, width or height specified\n");
    return -1;
  }

  if (libpath == NULL || outfile == NULL) {
    g_printerr ("Invalid library path or output file specified. Exiting\n");
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-ucx-test3-client-pipeline");

  ucxclientsrc = gst_element_factory_make ("nvdsucxclientsrc", "ucxclientsrc");
  caps_filter1 = gst_element_factory_make ("capsfilter", "caps-filter1");
  metaext = gst_element_factory_make ("nvdsmetaextract", "nvds-meta-extract");
  streamdemux = gst_element_factory_make ("nvstreamdemux", "nvdemux");
  audioconv = gst_element_factory_make ("audioconvert", "audio-convert");
  caps_filter2 = gst_element_factory_make ("capsfilter", "caps-filter2");
  waveenc = gst_element_factory_make ("wavenc", "wavenc");
  filesink = gst_element_factory_make ("filesink", "filesink");

  if (!ucxclientsrc || !caps_filter1 || !metaext || !streamdemux ||
      !audioconv || !caps_filter2 || !waveenc || !filesink) {
    g_printerr ("Failed to create some element. Exiting\n");
    return -1;
  }

  /* Set ucxclientsrc properties */
  g_object_set (G_OBJECT (ucxclientsrc), "addr", addr, "port", port,
      "nvbuf-batch-size", 1, "num-nvbuf", 4, "nvbuf-memory-type", 2,
      "buf-type", 1, NULL);

  caps1 = gst_caps_new_simple ("audio/x-raw",
      "format", G_TYPE_STRING, audformat1,
      "rate", G_TYPE_INT, 48000,
      "channels", G_TYPE_INT, 1, "layout", G_TYPE_STRING, "interleaved", NULL);

  feature = gst_caps_features_new ("memory:NVMM", NULL);
  gst_caps_set_features (caps1, 0, feature);
  g_object_set (G_OBJECT (caps_filter1), "caps", caps1, NULL);

  g_object_set (G_OBJECT (metaext), "deserialize-lib", libpath, NULL);

  g_object_set (G_OBJECT (filesink), "location", outfile, NULL);

  caps2 = gst_caps_new_simple ("audio/x-raw",
      "format", G_TYPE_STRING, audformat2, NULL);
  g_object_set (G_OBJECT (caps_filter2), "caps", caps2, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  gst_bin_add_many (GST_BIN (pipeline), ucxclientsrc, caps_filter1, metaext,
      streamdemux, audioconv, caps_filter2, waveenc, filesink, NULL);

  if (!gst_element_link_many (ucxclientsrc, caps_filter1, metaext,
          streamdemux, NULL)) {
    g_printerr ("Failed to link some elements till streamdemux. Exiting.\n");
    return -1;
  }

  GstPad *demuxsrcpad, *audsinkpad;
  gchar pad_name[16] = { };

  g_snprintf (pad_name, 15, "src_%u", 0);

  demuxsrcpad = gst_element_get_request_pad (streamdemux, pad_name);
  if (!demuxsrcpad) {
    g_printerr ("Failed to request src pad from demux. Exiting.\n");
    return -1;
  }

  audsinkpad = gst_element_get_static_pad (audioconv, "sink");
  if (!audsinkpad) {
    g_printerr ("Failed to get sink pad for audio converter. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (demuxsrcpad, audsinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link demux src and audio conv sink pads.\n");
    return -1;
  }

  gst_object_unref (demuxsrcpad);
  gst_object_unref (audsinkpad);

  if (!gst_element_link_many (audioconv, caps_filter2, waveenc, filesink, NULL)) {
    g_printerr ("Failed to link elements till filesink. Exiting.\n");
    return -1;
  }

  g_print ("Saving to file: %s\n", outfile);
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);

  return 0;
}
