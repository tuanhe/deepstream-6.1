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

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  g_print ("In cb_newpad\n");
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;

  /* Need to check if the pad created by the decodebin is for audio and not
   * video. */
  if (!strncmp (name, "audio", 5)) {
    /* Get the source bin ghost pad */
    GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
    if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
            decoder_src_pad)) {
      g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
    }
    gst_object_unref (bin_ghost_pad);
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  g_print ("Decodebin child added: %s\n", name);
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[16] = { };

  g_snprintf (bin_name, 15, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri,
      "async-handling", 1, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *ucxserversink = NULL, *audioconv = NULL,
      *caps_filter = NULL, *audiores = NULL, *absplit = NULL, *streammux = NULL,
      *metains = NULL;
  GstCaps *caps = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  gint64 port = 0;
  gchar *addr = NULL, *libpath = NULL, *endptr0 = NULL, *uri = NULL;
  gchar *audformat = "F32LE";

  int current_device = -1;
  cudaGetDevice (&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, current_device);

  /* Check input arguments */
  if (argc != 5) {
    g_printerr ("Usage: %s <addr> <port> "
        "<audio metadata serialization lib path> <uri1>\n", argv[0]);
    return -1;
  }

  addr = argv[1];
  port = g_ascii_strtoll (argv[2], &endptr0, 10);
  libpath = argv[3];
  uri = argv[4];

  if (!g_hostname_is_ip_address (addr)) {
    g_printerr ("Addr specified is not a valid IP address/hostname: %s\n",
        addr);
    return -1;
  }

  if ((port == 0 && endptr0 == argv[2]) || port <= 0) {
    g_printerr ("Incorrect port specified\n");
    return -1;
  }

  if (libpath == NULL || uri == NULL) {
    g_printerr ("Invalid library path or URI specified\n");
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("ds-ucx-test3-server-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");

  if (!pipeline || !streammux) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  gst_bin_add (GST_BIN (pipeline), streammux);

  GstElement *source_bin = create_source_bin (0, uri);
  if (!source_bin) {
    g_printerr ("Failed to create source bin. Exiting.\n");
    return -1;
  }

  gst_bin_add (GST_BIN (pipeline), source_bin);

  audioconv = gst_element_factory_make ("audioconvert", "audio-convert");
  caps_filter = gst_element_factory_make ("capsfilter", "caps_filter");
  audiores = gst_element_factory_make ("audioresample", ":audio-resample");
  absplit = gst_element_factory_make ("audiobuffersplit", "audio-buffer-split");
  metains = gst_element_factory_make ("nvdsmetainsert", "nvds-meta-insert");
  ucxserversink = gst_element_factory_make ("nvdsucxserversink", "serversink");

  if (!audioconv || !caps_filter || !audiores || !absplit ||
      !metains || !ucxserversink) {
    g_printerr ("Failed to create some elements. Exiting\n");
    return -1;
  }

  caps = gst_caps_new_simple ("audio/x-raw",
      "format", G_TYPE_STRING, audformat,
      "rate", G_TYPE_INT, 48000,
      "channels", G_TYPE_INT, 1, "layout", G_TYPE_STRING, "interleaved", NULL);
  g_object_set (G_OBJECT (caps_filter), "caps", caps, NULL);

  g_object_set (G_OBJECT (ucxserversink), "addr", addr, "port", port,
      "buf-type", 1, NULL);

  g_object_set (G_OBJECT (streammux), "max-latency", 250000000, NULL);

  g_object_set (G_OBJECT (metains), "serialize-lib", libpath, NULL);

  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  gst_bin_add_many (GST_BIN (pipeline), audioconv, caps_filter, audiores,
      absplit, metains, ucxserversink, NULL);

  GstPad *srcpad, *audsink;

  srcpad = gst_element_get_static_pad (source_bin, "src");
  if (!srcpad) {
    g_printerr ("Failed to get src pad of source bin. Exiting.\n");
    return -1;
  }

  audsink = gst_element_get_static_pad (audioconv, "sink");
  if (!audsink) {
    g_printerr ("Audioconvert static sink pad failed. Exiting.\n");
    return -1;
  }

  if (gst_pad_link (srcpad, audsink) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link source bin to audioconvert. Exiting.\n");
    return -1;
  }

  gst_object_unref (srcpad);
  gst_object_unref (audsink);

  if (!gst_element_link_many (audioconv, audiores, caps_filter, absplit, NULL)) {
    g_printerr ("Failed to link several elements with absplit. Exiting.\n");
    return -1;
  }

  GstPad *abssrcpad, *muxsinkpad;
  gchar pad_name[16] = { };

  abssrcpad = gst_element_get_static_pad (absplit, "src");
  if (!abssrcpad) {
    g_printerr ("Failed to get src pad for audiobuffersplit. Exiting\n");
    return -1;
  }

  g_snprintf (pad_name, 15, "sink_%u", 0);
  muxsinkpad = gst_element_get_request_pad (streammux, pad_name);
  if (!muxsinkpad) {
    g_printerr ("Failed to request sink pad for streammux. Exiting\n");
    return -1;
  }

  if (gst_pad_link (abssrcpad, muxsinkpad) != GST_PAD_LINK_OK) {
    g_printerr ("Failed to link buffer split src and streammux sink pads\n");
    return -1;
  }

  gst_object_unref (abssrcpad);
  gst_object_unref (muxsinkpad);

  if (!gst_element_link_many (streammux, metains, ucxserversink, NULL)) {
    g_printerr ("Failed to link elements with ucxserversink. Exiting.\n");
    return -1;
  }

  g_print ("Using URI: %s\n", uri);
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
