# Publishing the dashboard

## Deploying from a private repository

Streamlit Community Cloud can deploy from a **private** GitHub repository, and the
resulting app can still be made publicly viewable. Repository visibility and app
visibility are two separate settings:

| | Repository | Deployed app |
| --- | --- | --- |
| Who can read the code | you and collaborators | nobody |
| Who can open the URL | — | anyone, if you set the app to public |

So "private code, public demo" is a supported combination, and it is the one you
want for a reviewer-facing link.

### Steps

1. Sign in at <https://share.streamlit.io> with the GitHub account that owns the
   repository.
2. When it asks for GitHub permissions, **grant private-repository access**. The
   default sign-in only sees public repositories; if you skip this the repo will
   not appear in the picker. If you already signed in without it, revoke the
   Streamlit authorisation in GitHub → Settings → Applications and sign in again.
3. **Create app** → pick the repository, the branch, and set the main file path.
   If this directory is a subfolder of the repository, the path is e.g.
   `app/app.py`, not `app.py`.
4. Deploy. `requirements.txt` and `.streamlit/config.toml` are picked up
   automatically; there is nothing to configure.
5. In the app's **Settings → Sharing**, choose who can view it. Set it to public
   for a link you can put in a paper or a rebuttal.

Check these before you rely on the link, because the free-tier terms can change:
that private repositories are still included in your plan, and that app visibility
is set the way you intend.

### What a public app exposes

The code stays private, but everything the app *serves* becomes readable by
anyone with the URL:

- all 4,306 audit records, including questions, answers and scores;
- all 753 microscope crops;
- both documents that the Review log exports.

That is fine here — it is the material the paper describes and the data reviewers
were asked to be given. Just be deliberate about it: a public app is a public data
release even when the repository is private. If you would rather not publish the
data yet, set the app to private and invite reviewers by email address instead.

### Practical notes

- ~39 MB of crops plus a 216 KB record table. Well inside the limits; no Git LFS.
- Community Cloud sleeps idle apps and wakes them on the next visit. First load
  after a sleep takes a few seconds.
- `maxUploadSize` is set to 300 MB in `.streamlit/config.toml` so the
  "upload your own run" path works with a real archive of crops.
- The **Live model** tab cannot work on Community Cloud: it has no route into your
  network, so it cannot reach a checkpoint served on the GPU machine. Leave it
  unconfigured there and demonstrate it locally over an SSH tunnel
  ([RUN_ON_GPU.md](RUN_ON_GPU.md) §C). Do not open the serving host to the
  internet to work around this.

## Alternatives, if Community Cloud does not suit

- **Hugging Face Spaces** — supports Streamlit, private or public, and a private
  Space can be shared with named accounts. Reasonable if you want the data behind
  a login.
- **Self-hosted** — `streamlit run app.py --server.port 8501` behind a reverse
  proxy with authentication. The app has no login of its own, so do not expose it
  directly.
- **Screen-recorded walkthrough** — if a hosted demo is not wanted at all, a short
  recording plus the repository covers what reviewers asked for.

## Choosing a licence

Short version: **PolyForm Noncommercial 1.0.0** for the code, and keep the data
under the terms it already carries. Read the caveats below before committing.

### What a licence can and cannot protect

A licence covers the expression — this code, these documents, these figures. It
does **not** protect the idea. Nobody can be stopped by a licence from reading the
paper and building their own answer–action auditing tool; that is what
publication is for. What the paper gives you is priority and citation, not
exclusivity. If exclusivity over the method genuinely matters, that is a patent
question and needs your institution's technology-transfer office, ideally before
publication.

### Recommended: PolyForm Noncommercial 1.0.0

<https://polyformproject.org/licenses/noncommercial/1.0.0/>

- Anyone may read, run, modify and share the code for research, teaching and
  personal use.
- Commercial use is not permitted without a separate agreement with you.
- It is short, plainly written, and drafted by lawyers for exactly this purpose,
  so it is far safer than a hand-written restriction.

Trade-off to know about: it is **not** an open-source licence by the OSI
definition. For a workshop artifact whose purpose is reproducibility this is
normally accepted, but if a venue or funder requires OSI-approved open source, this
will not satisfy it.

Other options, if the fit is wrong:

| Licence | When to prefer it |
| --- | --- |
| **PolyForm Strict 1.0.0** | You want people to be able to read and verify the code but not build on it. |
| **BUSL-1.1** | You want a non-commercial restriction now that automatically becomes open source after a fixed period. |
| **Apache-2.0** | You would rather maximise adoption and citation, and accept commercial use. Includes an explicit patent grant. |
| **All rights reserved** | Simplest, but reviewers cannot legally run your code, which works against the reproducibility request. Not recommended here. |

### The data needs separate, careful treatment

This is the part that most needs checking, and it is not something I can settle
for you. The 753 crops in `release/images/` are derived from three public
datasets — **Lucchi**, **VNC** and **MitoEM** — each with its own licence. You
cannot place a derived work under terms that conflict with the upstream ones, in
either direction: you can neither grant more freedom than you were given, nor
necessarily add restrictions if a licence has a share-alike clause.

Before publishing:

1. Find the licence for each of the three datasets (MitoEM in particular has its
   own usage terms).
2. Confirm that redistributing 256×256 crops with derived centroid coordinates is
   permitted, and under what attribution requirement.
3. Record each dataset's licence and required attribution in `PROVENANCE.md`.

If any dataset turns out not to permit redistribution, the fix is small: ship the
records without the crops. The dashboard already handles missing images — it draws
the point geometry on a labelled blank grid and says the image is unavailable — so
the audit still works, only with less visual evidence.

### Files to add

Once you have decided:

- `LICENSE` — the chosen licence text, verbatim from its source.
- `CITATION.cff` — so the repository offers a citation; GitHub renders it as a
  "Cite this repository" button.
- A **Licence** section in `PROVENANCE.md` listing each dataset, its licence, and
  its attribution.

I have not added these files, because a licence is a legal declaration about your
work and the dataset terms need confirming first. Say which licence you want and I
will add all three.
