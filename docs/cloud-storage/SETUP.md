# Cloud Storage Setup

ClipAI can pull videos directly from Google Drive and Box. This guide
walks you through creating the OAuth applications each provider needs
and wiring them into the ClipAI container.

Skip straight to [Unraid template variables](#unraid-template-variables)
if you only need the env var names.

---

## 0. The encryption key (required)

OAuth refresh tokens are stored encrypted on disk using Fernet (AES-128
in CBC mode with HMAC). ClipAI needs a persistent key so your cloud
sessions survive a container restart. Generate one with:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Paste the output into the `CLIPAI_TOKEN_ENC_KEY` env var. **Do not
change it after the fact** — rotating this key invalidates every stored
cloud session and users will have to reconnect.

If you forget to set it, the app will still boot and generate an
ephemeral key, but a loud warning will appear in the logs and all your
stored cloud connections will evaporate the next time the container
restarts.

---

## 1. Google Drive

### Create the OAuth app

1. Go to [Google Cloud Console](https://console.cloud.google.com/) and
   create (or pick) a project.
2. In the left-hand menu: **APIs & Services → Library**. Enable
   **Google Drive API**.
3. **APIs & Services → OAuth consent screen**. Pick **External**,
   give the app any name (e.g. `ClipAI`), and add your email as a test
   user. You don't need to publish the app — personal OAuth projects
   can stay in "Testing" mode.
4. **APIs & Services → Credentials → Create Credentials → OAuth client
   ID**.
   - Application type: **Web application**
   - Authorized redirect URI: **the exact URL ClipAI is reachable at,
     followed by `/api/cloud/google_drive/callback`**. For a homelab
     Unraid install this is usually something like
     `http://192.168.1.20:8000/api/cloud/google_drive/callback`. If
     you're running behind a reverse proxy with HTTPS, use the HTTPS
     URL.
5. Click **Create**. Copy the **Client ID** and **Client Secret** —
   you'll paste them into the Unraid template in a moment.

### Scope

ClipAI requests only
`https://www.googleapis.com/auth/drive.readonly`. This is Google's
**read-only** scope — we cannot modify or delete files in your Drive.

### Env vars

```
GOOGLE_DRIVE_CLIENT_ID=<from step 5>
GOOGLE_DRIVE_CLIENT_SECRET=<from step 5>
GOOGLE_DRIVE_REDIRECT_URI=http://<your-clipai-host>/api/cloud/google_drive/callback
```

The redirect URI must match **exactly** what you pasted into the Google
console — including scheme, host, port, and path.

### "App unverified" screen

Because the project stays in Testing mode, Google shows an "unverified
app" warning the first time you sign in. Click **Advanced → Go to
ClipAI (unsafe)**. It's your own OAuth app talking to your own drive;
the warning exists because Google hasn't reviewed it. Publishing /
verification isn't necessary for a single-user homelab install.

---

## 2. Box

### Create the OAuth app

1. Go to the [Box Developer Console](https://app.box.com/developers/console)
   and click **Create New App**.
2. Pick **Custom App** → **OAuth 2.0 (User Authentication)**.
3. Give it a name (e.g. `ClipAI`). Continue.
4. On the app's **Configuration** tab:
   - **OAuth 2.0 Redirect URI**: add
     `http://<your-clipai-host>/api/cloud/box/callback`. As with Google
     Drive, this must match the `BOX_REDIRECT_URI` env var exactly.
   - **Application Scopes**: enable only
     **"Read all files and folders stored in Box"**. Leave every write
     scope unchecked.
5. Copy the **Client ID** and **Client Secret** from the **OAuth 2.0
   Credentials** section.

### Scope

ClipAI asks for the default user scope. What the user actually sees
depends on the app's **Configuration → Application Scopes** matrix —
leaving only "Read all files and folders" enabled is what makes the
integration read-only in practice.

### Env vars

```
BOX_CLIENT_ID=<from step 5>
BOX_CLIENT_SECRET=<from step 5>
BOX_REDIRECT_URI=http://<your-clipai-host>/api/cloud/box/callback
```

### Refresh-token rotation gotcha

Box is unusual in that it **invalidates the old refresh token every
time you exchange one for a new access token**. The ClipAI backend
handles this automatically — `backend/app/cloud/providers/box.py`
always persists the new refresh token returned from the refresh
response. You don't need to do anything, but it's worth knowing about
if you ever see a `401 invalid_grant` error after a long period of
inactivity: the user just needs to disconnect and reconnect in
Settings.

---

## 3. Connecting an account in ClipAI

Once the env vars are set and the container is restarted:

1. Go to **Settings → Advanced → Cloud Storage**.
2. You should see cards for Google Drive and Box. Configured providers
   show a blue **Connect** button; unconfigured ones show
   "Not configured" with a link back to this doc.
3. Click **Connect**. A new tab opens to the provider's OAuth screen.
4. After you approve the request, you'll be redirected back to
   `/settings?connected=<provider>`. The cloud storage card will now
   show **Connected as `<your-email>`**.

If nothing happens when you click Connect, or you land on an error
page, check:

- **Redirect URI mismatch.** The single most common failure. The URL
  the provider redirects to must be byte-for-byte identical to what
  you pasted into their console. Watch for `http` vs `https`, trailing
  slashes, and port numbers.
- **CLIPAI_CLOUD_STORAGE_ENABLED is not false.** If the feature flag
  is off the `/api/cloud/*` endpoints return 503 and the Settings
  section says "Cloud storage is disabled".
- **Container logs.** Token exchange errors get logged with full
  context; actual token values are scrubbed by the redaction filter in
  `backend/app/cloud/logging_filter.py` so it's safe to share the
  output when asking for help.

---

## 4. Using the picker

On the **Upload** page you'll see a new tab strip above the drop zone:

```
[ Local file ]  [ Google Drive ]  [ Box ]
```

The cloud tabs only appear for providers you've configured on the
server. Clicking one opens a folder browser with a search box. Select
a video and ClipAI streams it from the cloud directly into the same
analysis pipeline a local upload would go through — the file never
passes through your browser, and the ClipAI container never stores a
raw access token on disk.
