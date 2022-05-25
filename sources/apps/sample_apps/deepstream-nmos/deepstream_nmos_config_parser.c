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


#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "deepstream_nmos_config_parser.h"
#include "deepstream_common.h"

static gchar*
get_absolute_path (const gchar *srcPath, const gchar* cfgFilePath)
{
  gchar *absFilePath = NULL;
  if (srcPath && srcPath[0] == '/')
    return g_strdup (srcPath);

  char *absPath = realpath (cfgFilePath, NULL);
  if (!absPath) {
    NVGSTDS_ERR_MSG_V ("Error in getting absolute path of %s", cfgFilePath);
    return NULL;
  }

  if (!srcPath) {
    absFilePath = g_strdup (absPath);
    free (absPath);
    return absFilePath;
  }

  gchar *delim = g_strrstr (absPath, "/");
  *(delim + 1) = '\0';

  if (g_str_has_prefix (srcPath, "file://")) {
    absFilePath = g_strconcat (absPath, srcPath + 7, NULL);
  } else {
    absFilePath = g_strconcat (absPath, srcPath, NULL);
  }
  free (absPath);
  return absFilePath;
}

gboolean
parse_gie (NvDsNmosAppConfig *appConfig, GKeyFile *keyFile, gchar *group,
      gchar *cfgFilePath)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  if (g_key_file_get_integer (keyFile, group, CONFIG_GROUP_ENABLE, &error) == FALSE) {
    appConfig->enablePgie = FALSE;
    return TRUE;
  }

  keys = g_key_file_get_keys (keyFile, group, NULL, &error);
  if (error) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    goto done;
  }

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      appConfig->enablePgie = g_key_file_get_boolean (keyFile, group,
              CONFIG_GROUP_ENABLE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_PGIE_CONFIG_FILE)) {
      gchar *pgieConfFile = g_key_file_get_string (keyFile, group,
              CONFIG_GROUP_PGIE_CONFIG_FILE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }

      gchar *absPath = get_absolute_path (pgieConfFile, cfgFilePath);
      if (!absPath) {
        NVGSTDS_ERR_MSG_V ("Error in getting absolute path of %s", pgieConfFile);
        g_free (pgieConfFile);
        goto done;
      }
      appConfig->pgieConfFile = absPath;
      g_free (pgieConfFile);
    }
  }
  ret = TRUE;

done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  return ret;
}

gboolean
parse_app (NvDsNmosAppConfig *appConfig, GKeyFile *keyFile, gchar *group,
      gchar *cfgFilePath)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;

  keys = g_key_file_get_keys (keyFile, group, NULL, &error);
  if (error) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    goto done;
  }

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_APP_SEED)) {
      appConfig->seed = g_key_file_get_string (keyFile, group,
              CONFIG_GROUP_APP_SEED, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_HOST_NAME)) {
      appConfig->hostName = g_key_file_get_string (keyFile, group,
              CONFIG_GROUP_APP_HOST_NAME, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_APP_HTTP_PORT)) {
      appConfig->httpPort = g_key_file_get_integer (keyFile, group,
              CONFIG_GROUP_APP_HTTP_PORT, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else {
      NVGSTDS_ERR_MSG_V ("Unknown key %s for group %s", *key, group);
    }
  }

  ret = TRUE;

done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  return ret;
}

gboolean
parse_sender (NvDsNmosSinkConfig *sinkConfig, GKeyFile *keyFile, gchar *group,
      gchar *cfgFilePath)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  static GList *sinkList = NULL;

  keys = g_key_file_get_keys (keyFile, group, NULL, &error);
  if (error) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    goto done;
  }

  if (!g_key_file_get_integer (keyFile, group, CONFIG_GROUP_ENABLE, &error)) {
    // group is not enabled, no need to parse further.
    return TRUE;
  }

  gchar *idStartPtr = group + strlen (CONFIG_GROUP_SENDER);
  gchar *idEndPtr = NULL;
  guint sinkIndex = g_ascii_strtoull (idStartPtr, &idEndPtr, 10);

  if (idStartPtr == idEndPtr || *idEndPtr != '\0') {
    NVGSTDS_ERR_MSG_V ("Sink group \"[%s]\" is not in the form \"[sink<%%d>]\"",
        group);
    return FALSE;
  }

  // Check if a sender with same id has already been parsed.
  if (g_list_find (sinkList, GUINT_TO_POINTER(sinkIndex)) != NULL) {
      NVGSTDS_ERR_MSG_V ("Did not parse sender group \"[%s]\". Another sender group"
          " with id %d already exists", group, sinkIndex);
      return FALSE;
  }

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      sinkConfig->enable = g_key_file_get_boolean (keyFile, group,
              CONFIG_GROUP_ENABLE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SDPFILE)) {
      gchar *sdpFile = g_key_file_get_string (keyFile, group,
              CONFIG_GROUP_SDPFILE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
      gchar *absPath = get_absolute_path (sdpFile, cfgFilePath);
      if (!absPath) {
        NVGSTDS_ERR_MSG_V ("Error in getting absolute path of %s", sdpFile);
        g_free (sdpFile);
        goto done;
      }
      sinkConfig->sdpFile = absPath;
      g_free (sdpFile);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TYPE)) {
      sinkConfig->type = g_key_file_get_integer (keyFile, group,
              CONFIG_GROUP_TYPE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_PAYLOAD_SIZE)) {
      sinkConfig->payloadSize = g_key_file_get_integer (keyFile, group,
              CONFIG_GROUP_SINK_PAYLOAD_SIZE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_PACKET_PER_LINE)) {
      sinkConfig->packetsPerLine = g_key_file_get_integer (keyFile, group,
              CONFIG_GROUP_SINK_PACKET_PER_LINE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else {
      NVGSTDS_ERR_MSG_V ("Unknown key %s for group %s", *key, group);
    }
  }

  sinkList = g_list_prepend (sinkList, GUINT_TO_POINTER(sinkIndex));
  sinkConfig->index = sinkIndex;
  ret = TRUE;

done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  return ret;
}

gboolean
parse_receiver (NvDsNmosSrcConfig *srcConfig, GKeyFile *keyFile, gchar *group,
      gchar *cfgFilePath)
{
  gboolean ret = FALSE;
  gchar **keys = NULL;
  gchar **key = NULL;
  GError *error = NULL;
  static GList *sourceList = NULL;

  keys = g_key_file_get_keys (keyFile, group, NULL, &error);
  if (error) {
    NVGSTDS_ERR_MSG_V ("%s", error->message);
    goto done;
  }

  if (!g_key_file_get_integer (keyFile, group, CONFIG_GROUP_ENABLE, &error)) {
    // group is not enabled, no need to parse further.
    return TRUE;
  }

  gchar *idStartPtr = group + strlen(CONFIG_GROUP_RECEIVER);
  gchar *idEndPtr = NULL;
  guint srcIndex = g_ascii_strtoull (idStartPtr, &idEndPtr, 10);

  if (idStartPtr == idEndPtr || *idEndPtr != '\0') {
    NVGSTDS_ERR_MSG_V ("Source group \"[%s]\" is not in the form \"[source<%%d>]\"",
        group);
    return FALSE;
  }

  // Check if a receiver with same id has already been parsed.
  if (g_list_find (sourceList, GUINT_TO_POINTER(srcIndex)) != NULL) {
      NVGSTDS_ERR_MSG_V ("Did not parse receiver group \"[%s]\". Another receiver group"
          " with id %d already exists", group, srcIndex);
      return FALSE;
  }

  for (key = keys; *key; key++) {
    if (!g_strcmp0 (*key, CONFIG_GROUP_ENABLE)) {
      srcConfig->enable = g_key_file_get_boolean (keyFile, group,
              CONFIG_GROUP_ENABLE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SDPFILE)) {
      gchar *sdpFile = g_key_file_get_string (keyFile, group,
              CONFIG_GROUP_SDPFILE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
      gchar *absPath = get_absolute_path (sdpFile, cfgFilePath);
      if (!absPath) {
        NVGSTDS_ERR_MSG_V ("Error in getting absolute path of %s", sdpFile);
        g_free (sdpFile);
        goto done;
      }
      srcConfig->sdpFile = absPath;
      g_free (sdpFile);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_TYPE)) {
      srcConfig->type = g_key_file_get_integer (keyFile, group,
              CONFIG_GROUP_TYPE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_SDPFILE)) {
      gchar *sdpFile = g_key_file_get_string (keyFile, group,
              CONFIG_GROUP_SINK_SDPFILE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
      gchar *absPath = get_absolute_path (sdpFile, cfgFilePath);
      if (!absPath) {
        NVGSTDS_ERR_MSG_V ("Error in getting absolute path of %s", sdpFile);
        g_free (sdpFile);
        goto done;
      }
      srcConfig->sinkSdpFile = absPath;
      g_free (sdpFile);
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_TYPE)) {
      srcConfig->sinkType = g_key_file_get_integer (keyFile, group,
              CONFIG_GROUP_SINK_TYPE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_PAYLOAD_SIZE)) {
      srcConfig->payloadSize = g_key_file_get_integer (keyFile, group,
              CONFIG_GROUP_SINK_PAYLOAD_SIZE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else if (!g_strcmp0 (*key, CONFIG_GROUP_SINK_PACKET_PER_LINE)) {
      srcConfig->packetsPerLine = g_key_file_get_integer (keyFile, group,
              CONFIG_GROUP_SINK_PACKET_PER_LINE, &error);
      if (error) {
        NVGSTDS_ERR_MSG_V ("%s", error->message);
        goto done;
      }
    } else {
      NVGSTDS_ERR_MSG_V ("Unknown key %s for group %s", *key, group);
    }
  }

  if (srcConfig->sinkType && !srcConfig->sinkSdpFile) {
    NVGSTDS_ERR_MSG_V ("sink-sdp-file not provided for group %s", group);
    goto done;
  }

  sourceList = g_list_prepend (sourceList, GUINT_TO_POINTER(srcIndex));
  srcConfig->index = srcIndex;
  ret = TRUE;

done:
  if (error) {
    g_error_free (error);
  }
  if (keys) {
    g_strfreev (keys);
  }
  return ret;
}

gboolean parse_config_file (NvDsNmosAppCtx *appCtx, gchar *cfgFilePath)
{
  NvDsNmosAppConfig *config = &appCtx->config;
  GError *error = NULL;
  gboolean ret = FALSE;
  gchar **groups = NULL;
  gchar **group;

  config->numSrc = 0;
  config->numSink = 0;

  GKeyFile *cfgFile = g_key_file_new ();
  if (!g_key_file_load_from_file (cfgFile, cfgFilePath, G_KEY_FILE_NONE,
          &error)) {
    NVGSTDS_ERR_MSG_V ("Failied to load uri file: %s", error->message);
    goto done;
  }

  groups = g_key_file_get_groups (cfgFile, NULL);
  for (group = groups; *group; group++) {
    if (g_str_has_prefix (*group, CONFIG_GROUP_RECEIVER)) {
      ret = parse_receiver (&config->srcConfigs[config->numSrc], cfgFile, *group, cfgFilePath);
      if (!ret) {
        NVGSTDS_ERR_MSG_V ("Failed to parse %s", *group);
        goto done;
      }
      if (config->srcConfigs[config->numSrc].enable)
        config->numSrc++;
    } else if (g_str_has_prefix (*group, CONFIG_GROUP_SENDER)) {
      ret = parse_sender (&config->sinkConfigs[config->numSink], cfgFile, *group, cfgFilePath);
      if (!ret) {
        NVGSTDS_ERR_MSG_V ("Failed to parse %s", *group);
        goto done;
      }
      if (config->sinkConfigs[config->numSink].enable)
        config->numSink++;
    } else if (!g_strcmp0 (*group, CONFIG_GROUP_APP)) {
      ret = parse_app (config, cfgFile, *group, cfgFilePath);
      if (!ret) {
        NVGSTDS_ERR_MSG_V ("Failed to parse %s", *group);
        goto done;
      }
    } else if (!g_strcmp0 (*group, CONFIG_GROUP_PGIE)) {
      ret = parse_gie (config, cfgFile, *group, cfgFilePath);
      if (!ret) {
        NVGSTDS_ERR_MSG_V ("Failed to parse %s", *group);
        goto done;
      }
    } else {
      NVGSTDS_WARN_MSG_V ("Unknown group %s", *group);
    }
  }
  ret = TRUE;

done:
  if (cfgFile) {
    g_key_file_free (cfgFile);
  }

  if (groups) {
    g_strfreev (groups);
  }

  if (error) {
    g_error_free (error);
  }
  return ret;
}
