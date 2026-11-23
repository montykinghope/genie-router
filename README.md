# Genie Router Agent

Routes a question to the right Genie agent, and records on every request which Genie
conversation it created, so you can match a router request to Genie's own history.

It deploys as a Databricks App serving `POST /api/invocations`, callable from your own front
end. A browser chat page is included for testing.

Setup is done entirely in the Databricks workspace UI. No command line, no local install.
Allow about 30 minutes.

**Before you start you need:** two or more Genie agents already built, and permission to create
a Databricks App, an MLflow experiment, and to write to a Unity Catalog schema.

---

## 1. Upload the code

1. In the workspace sidebar, open **Workspace**, go to your user folder, and click
   **Create > Folder**. Name it **genie-router**.
2. Open that folder, click **Create > Upload files**, and upload everything from this package,
   keeping the folder structure: **app.yaml**, **pyproject.toml**, **agent.toml**, and the
   **agent/**, **runtime/** and **ui/** folders.

## 2. Point the router at your Genie agents

Get each Genie agent's **space ID** from its URL, the long value after `/genie/rooms/`:

```
https://<your-workspace>/genie/rooms/01f0a1b2c3d4e5f60718293a4b5c6d7e
```

Open **genie-router > agent > tools > genie_router.py** and edit the `GENIE_AGENTS` list. There
are two example entries; add more by copying one. For each agent set:

- **name** — the identifier the model uses to call this tool. Letters, digits, underscores and
  hyphens only, **no spaces**. Mirroring the Genie agent's own name works well, for example
  `nyc_taxi_trips`.
- **space\_id** — the ID from the URL above.
- **description** — what that Genie agent's data covers. **This is what the routing decision is
  based on**, so be specific. Name the entities and the kinds of question it can answer, and
  what it can't. "Sales data" routes poorly. "Revenue by region, product or period; order
  volumes and status; pipeline stages" routes well.

Save, then open **agent > agent.py** and update the list inside `SYSTEM_PROMPT` so it names the
same agents:

```
You have these Genie agents, each covering a different dataset:
- nyc_taxi_trips: taxi pickups, dropoffs, fares and trip distances.
- bakehouse_sales: franchise sales, transactions, products and store performance.
```

Leave the numbered rules below that list as they are. They handle questions that span two
agents or that no agent covers.

## 3. Create an experiment for the traces

1. Sidebar > **Experiments** > **Create experiment**.
2. **Experiment name:** enter a plain name with no slashes, for example `genie-router-traces`.
3. **Trace storage location:** choose **Unity Catalog**.
4. **Unity Catalog location:** pick a catalog and schema you can write to.
5. **Table name prefix:** enter `genie_router`.
6. Click **Create experiment**, then copy the ID number from the end of the browser URL.

## 4. Add the experiment ID

Open **genie-router > app.yaml** and replace `PASTE_YOUR_EXPERIMENT_ID_HERE` with that number,
keeping the quote marks. Save.

## 5. Create the app

1. Sidebar > **Compute > Apps** (or the app switcher > **Apps**) > **Create app**.
2. Choose **Create a custom app**, name it **genie-router**, and click through to **Create app**.
3. On the app's page, copy the **service principal ID**. It is a UUID like
   `3f8e21c7-9b44-4a1e-8d52-6ac0fe71b204`. You need it three times below. When granting
   permissions, search by this UUID rather than the display name.

## 6. Grant the app access

The app runs as its own identity, so grant it three things:

| Where | Permission |
| --- | --- |
| The app's **App resources** > **Add resource** > MLflow experiment | **Can Edit** on the experiment from step 3 |
| **Catalog** > your schema from step 3 > **Permissions** | **USE SCHEMA**, **SELECT**, **MODIFY** |
| Each Genie agent > **Share** | **Can Run** |

## 7. Deploy

On the app's page click **Deploy**, select your **genie-router** folder, and click **Deploy**.
This takes a few minutes. Wait until the app reports that it is running.

## 8. Test it

Open the app's URL. You get a chat page. Ask:

- A clear question for each Genie agent. Each should go to the right one, and the reply names
  which agent answered.
- A question that could belong to either. This is how you find descriptions that need
  sharpening.
- Something no agent covers, such as "what is the weather today?". It should decline and say
  what it does cover.

Follow-up questions work within a conversation.

If a question goes to the wrong agent, improve that agent's **description** in step 2 and
deploy again.

---

## Reading the traces

Every request writes one trace recording the question, the agent chosen, the SQL Genie ran, the
answer, and the Genie conversation and message IDs.

**In the browser:** Sidebar > **Experiments** > your experiment > **Traces**.

**In SQL**, using the schema and prefix from step 3. This returns one row per Genie call:

```sql
SELECT
  t.request_time,
  call.tool            AS routed_to,
  call.conversation_id AS conversation_id,
  call.message_id      AS message_id
FROM my_catalog.my_schema.genie_router_trace_unified t
LATERAL VIEW explode(
  from_json(t.tags['genie.calls'],
    'array<struct<tool:string,space_id:string,conversation_id:string,message_id:string>>')
) AS call
WHERE t.tags['invoked_by'] = 'genie-router'
ORDER BY t.request_time DESC;
```

`conversation_id` matches Genie's own conversation records, so you can join this to Genie
history to see which conversations came from the router.

The experiment creates several tables. **`<prefix>_trace_unified`** is the one to query; the
`_otel_*` tables hold the same data in a nested format.

---

## Calling it from your own application

The chat page is optional. The app's interface is a single endpoint:

```
POST https://<your-app-url>/api/invocations
{
  "id": "<a new UUID per request>",
  "session_id": "<a stable UUID per conversation>",
  "input": {"messages": [{"role": "user", "content": "Revenue by region last quarter?"}]}
}
```

Reuse `session_id` across turns for follow-up questions. The reply is at `output.output`, a
list of messages: the last one with text content is the answer, and earlier ones carry the tool
call naming the Genie agent chosen. Add `"stream": true` for server-sent events, or
`"background": true` to poll a status URL for long questions. Authentication is OAuth; personal
access tokens are not accepted by Databricks Apps.

To run API-only, delete the `ENABLE_CHAT_UI` lines from **app.yaml** and deploy again.

---

## If something goes wrong

Open the app's **Logs** tab.

| Symptom | Fix |
| --- | --- |
| Questions fail with a permissions error naming a space ID | The Genie **Can Run** grant in step 6 |
| No traces appear | Check the experiment ID in **app.yaml**, and the experiment and schema grants in step 6 |
| App won't start | Usually a typo in **app.yaml** or **genie_router.py**. A healthy start ends with `Application startup complete` |
| An error naming a placeholder space ID | A `GENIE_AGENTS` entry in step 2 still has its example value |
| An error about a tool name | A `name` in step 2 contains a space or other invalid character |

## Adding a Genie agent later

Add an entry to `GENIE_AGENTS` in **genie_router.py**, add a matching line to `SYSTEM_PROMPT` in
**agent.py**, grant the app **Can Run** on the new Genie agent, and deploy again.
