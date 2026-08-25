# Interactive session title

State what this session is about, where it currently stands, and what the human should do next.

## Session

<details class="card" data-tressoir-markdown>
  <summary>
    <span class="card-title">Kickoff — what we aligned on</span>
    <span class="card-oneliner">Summarize the settled context.</span>
    <span class="card-badge">Answered</span>
  </summary>

Add the concise recap here.

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">Latest — what needs your call</span>
    <span class="card-oneliner">State the decision needed now.</span>
    <span class="card-badge">Awaiting</span>
  </summary>

Explain the relevant evidence and tradeoffs in simple, visible prose.

<article class="decision" data-tressoir-decision data-decision-state="unresolved"
  aria-labelledby="session-proceed-question">
  <header class="decision-header">
    <div>
      <h3 class="decision-title" id="session-proceed-question">How should this proceed?</h3>
      <p class="decision-context">Check every applicable answer. Choosing at least one marks the decision resolved.</p>
    </div>
    <span class="decision-state" data-decision-indicator role="status" aria-live="polite">Unresolved</span>
  </header>
  <fieldset class="decision-options">
    <legend class="visually-hidden">Decision answers</legend>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="session.proceed.accept">
      <span><strong>Proceed as proposed</strong><small>Use the current recommendation.</small></span>
    </label>
    <label class="decision-option">
      <input type="checkbox" data-tressoir-input="session.proceed.revise">
      <span><strong>Revise first</strong><small>Describe the change below.</small></span>
    </label>
  </fieldset>
  <div class="field decision-feedback">
    <label for="session-proceed-response">Free Response</label>
    <textarea id="session-proceed-response" rows="2" data-tressoir-input="session.proceed.feedback"
      data-tressoir-autogrow="2:6" placeholder="Add anything the choices miss…"></textarea>
  </div>
</article>

</details>
