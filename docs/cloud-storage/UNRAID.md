# Unraid template variables for Cloud Storage

The ClipAI Community Applications template needs the following new
variables to expose the cloud-storage integration to operators. Add
them as **Variables** (not Ports / Paths) in the template XML.

If you're hand-editing the existing template (usually
`/boot/config/plugins/dockerMan/templates-user/my-ClipAI.xml`), append
the entries below inside the existing `<Config>` block section.

> **TODO** — the Community Applications template is maintained
> separately from this repo. When this branch lands, the next template
> bump should add every variable below. Until then, operators can set
> these directly under the container's **Edit → Advanced view → Add
> another Path, Port, Variable, Label or Device** menu.

---

## Required

Only `CLIPAI_TOKEN_ENC_KEY` is required for cloud storage. Every other
variable is optional — the app starts fine without them and simply
hides the cloud tabs when no provider is configured.

### `CLIPAI_TOKEN_ENC_KEY`

| Field        | Value                                                            |
| ------------ | ---------------------------------------------------------------- |
| Name         | Token Encryption Key                                              |
| Key          | `CLIPAI_TOKEN_ENC_KEY`                                            |
| Description  | Fernet key for encrypting stored OAuth refresh tokens at rest. Generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. **Do not change after initial setup — rotating invalidates all stored cloud sessions.** |
| Default      | *(blank — user must fill in)*                                     |
| Mode         | Password (sensitive)                                              |
| Required     | Only if `CLIPAI_CLOUD_STORAGE_ENABLED` is true (the default)     |

---

## Optional — feature flag

### `CLIPAI_CLOUD_STORAGE_ENABLED`

| Field       | Value                                                             |
| ----------- | ----------------------------------------------------------------- |
| Name        | Cloud Storage Enabled                                              |
| Key         | `CLIPAI_CLOUD_STORAGE_ENABLED`                                     |
| Description | Master switch for the cloud storage integration. Set to `false` to kill all /api/cloud/* endpoints and hide the picker in the UI — useful as an emergency cutoff if something misbehaves in production. |
| Default     | `true`                                                             |

---

## Optional — Google Drive

### `GOOGLE_DRIVE_CLIENT_ID`

| Field       | Value                                                                |
| ----------- | -------------------------------------------------------------------- |
| Name        | Google Drive Client ID                                                |
| Key         | `GOOGLE_DRIVE_CLIENT_ID`                                             |
| Description | OAuth client id from Google Cloud Console. See docs/cloud-storage/SETUP.md. Leave blank to hide the Google Drive tab. |
| Default     | *(blank)*                                                             |

### `GOOGLE_DRIVE_CLIENT_SECRET`

| Field       | Value                                                               |
| ----------- | ------------------------------------------------------------------- |
| Name        | Google Drive Client Secret                                           |
| Key         | `GOOGLE_DRIVE_CLIENT_SECRET`                                        |
| Description | OAuth client secret paired with `GOOGLE_DRIVE_CLIENT_ID`.            |
| Default     | *(blank)*                                                            |
| Mode        | Password (sensitive)                                                 |

### `GOOGLE_DRIVE_REDIRECT_URI`

| Field       | Value                                                                |
| ----------- | -------------------------------------------------------------------- |
| Name        | Google Drive Redirect URI                                            |
| Key         | `GOOGLE_DRIVE_REDIRECT_URI`                                         |
| Description | Must match exactly what you pasted into the Google OAuth consent screen. For Unraid this is usually `http://<UNRAID-IP>:<PORT>/api/cloud/google_drive/callback`. |
| Default     | `http://localhost:8000/api/cloud/google_drive/callback`              |

---

## Optional — Box

### `BOX_CLIENT_ID`

| Field       | Value                                                                |
| ----------- | -------------------------------------------------------------------- |
| Name        | Box Client ID                                                         |
| Key         | `BOX_CLIENT_ID`                                                       |
| Description | OAuth client id from the Box Developer Console. Leave blank to hide the Box tab. |
| Default     | *(blank)*                                                             |

### `BOX_CLIENT_SECRET`

| Field       | Value                                                                |
| ----------- | -------------------------------------------------------------------- |
| Name        | Box Client Secret                                                     |
| Key         | `BOX_CLIENT_SECRET`                                                  |
| Description | OAuth client secret paired with `BOX_CLIENT_ID`.                      |
| Default     | *(blank)*                                                             |
| Mode        | Password (sensitive)                                                  |

### `BOX_REDIRECT_URI`

| Field       | Value                                                                |
| ----------- | -------------------------------------------------------------------- |
| Name        | Box Redirect URI                                                      |
| Key         | `BOX_REDIRECT_URI`                                                   |
| Description | Must match exactly what you pasted into the Box app's Configuration → OAuth 2.0 Redirect URIs. For Unraid this is usually `http://<UNRAID-IP>:<PORT>/api/cloud/box/callback`. |
| Default     | `http://localhost:8000/api/cloud/box/callback`                        |

---

## What happens with zero providers configured

The app boots normally. `/api/cloud/providers` returns an empty list
(or a list of providers with `configured: false`). The Upload page
shows only the **Local file** tab. Settings → Advanced → Cloud Storage
shows both provider cards in a "Not configured" state with a link to
the setup guide. No background work runs and no outbound calls are
made.

## Common problems

- **Operator never set `CLIPAI_TOKEN_ENC_KEY` and noticed cloud
  sessions vanish on every reboot.** Set the key in the template and
  restart once. Users will reconnect one more time and then the
  sessions will stick.
- **OAuth redirect never comes back.** The redirect URI in the
  template doesn't match what the operator pasted into the provider
  console. They must match byte-for-byte (scheme + host + port +
  path).
