(function () {
  "use strict";

  const elements = {
    answer: document.getElementById("answer-region"),
    alert: document.getElementById("global-alert"),
    count: document.getElementById("question-count"),
    duration: document.getElementById("request-duration"),
    sourceCount: document.getElementById("source-count"),
    sources: document.getElementById("sources-region"),
    evidenceCount: document.getElementById("evidence-count"),
    evidence: document.getElementById("evidence-region"),
    webStatus: document.getElementById("web-status"),
    web: document.getElementById("web-region"),
    evaluation: document.getElementById("evaluation-region"),
    benchmark: document.getElementById("benchmark-region"),
    documents: document.getElementById("documents-region"),
    submit: document.getElementById("ask-submit"),
    buttonLabel: document.querySelector("#ask-submit .button-label"),
    spinner: document.querySelector("#ask-submit .spinner-border"),
    stages: {
      retrieval: document.getElementById("stage-retrieval"),
      grading: document.getElementById("stage-grading"),
      routing: document.getElementById("stage-routing"),
      answer: document.getElementById("stage-answer"),
    },
    timings: {
      retrieval: document.getElementById("timing-retrieval"),
      grading: document.getElementById("timing-grading"),
      routing: document.getElementById("timing-routing"),
      answer_generation: document.getElementById("timing-answer_generation"),
    },
    liveDot: document.getElementById("live-dot"),
    liveStatus: document.getElementById("live-status"),
    readyDot: document.getElementById("ready-dot"),
    readyStatus: document.getElementById("ready-status"),
    statusUpdated: document.getElementById("status-updated"),
  };

  function setLoading(isLoading) {
    elements.submit.disabled = isLoading;
    elements.buttonLabel.textContent = isLoading ? "Running..." : "Run retrieval";
    elements.spinner.classList.toggle("d-none", !isLoading);
  }

  function showAlert(message) {
    elements.alert.textContent = message;
    elements.alert.classList.remove("d-none");
  }

  function clearAlert() {
    elements.alert.textContent = "";
    elements.alert.classList.add("d-none");
  }

  function setStage(name, value) {
    if (elements.stages[name]) {
      elements.stages[name].textContent = value;
    }
  }

  function resetStages() {
    Object.keys(elements.stages).forEach(function (name) {
      setStage(name, "Not reported");
    });
  }

  function renderEvidence(response) {
    const evidence = Array.isArray(response.retrieved_evidence) ? response.retrieved_evidence : [];
    const grading = Array.isArray(response.grading_decisions) ? response.grading_decisions : [];
    elements.evidenceCount.textContent = String(response.retrieved_chunk_count || evidence.length) + " chunks";
    if (!evidence.length) {
      elements.evidence.className = "evidence-list empty-state compact";
      elements.evidence.innerHTML = "<p class=\"mb-1\">No evidence reported</p><span>The workflow returned no retrieved chunks.</span>";
      return;
    }
    elements.evidence.className = "evidence-list";
    elements.evidence.innerHTML = evidence.map(function (item, index) {
      const decision = grading[index];
      const score = typeof item.retrieval_score === "number"
        ? " · score " + item.retrieval_score.toFixed(3)
        : "";
      const grade = decision ? decision.decision.toUpperCase() : "Not reported";
      return "<div class=\"evidence-item\"><div class=\"d-flex justify-content-between gap-2\">" +
        "<strong>" + escapeHtml(item.source_id || "Unidentified source") + "</strong>" +
        "<span class=\"hint\">" + escapeHtml(grade + score) + "</span></div>" +
        "<p>" + escapeHtml(item.text || "") + "</p>" +
        (decision && decision.rationale ? "<span class=\"hint\">Grading: " + escapeHtml(decision.rationale) + "</span>" : "") +
        "</div>";
    }).join("");
  }

  function renderWebStatus(web) {
    const value = web || {};
    const status = value.status || "not_attempted";
    const labels = {
      not_attempted: "Not attempted",
      disabled: "Disabled",
      succeeded: "Succeeded",
      failed: "Provider failed",
    };
    elements.webStatus.textContent = labels[status] || "Not reported";
    const enabled = value.enabled ? "enabled" : "disabled";
    const attempted = value.attempted ? "attempted" : "not attempted";
    elements.web.className = "empty-state compact";
    elements.web.innerHTML = "<p class=\"mb-1\">" + escapeHtml(labels[status] || "Not reported") +
      "</p><span>Search is " + enabled + " and was " + attempted + ". Sources returned: " +
      String(Number.isInteger(value.source_count) ? value.source_count : 0) + ".</span>";
  }

  function renderEvaluation(report, benchmark) {
    if (!report && !benchmark) {
      elements.evaluation.innerHTML = "<span class=\"placeholder-icon\" aria-hidden=\"true\">∅</span><p class=\"mb-1\">Evaluation data unavailable</p><span>No validated report is currently available.</span>";
      renderBenchmark(null);
      return;
    }
    const metrics = report && report.metrics ? Object.keys(report.metrics).map(function (key) {
      return escapeHtml(key) + ": " + escapeHtml(report.metrics[key]);
    }).join(" · ") : "No summary metrics";
    const cases = report && Array.isArray(report.cases) ? report.cases.length : 0;
    const benchmarkVariants = benchmark && benchmark.prompt_variants ? Object.keys(benchmark.prompt_variants).length : 0;
    const benchmarkTime = benchmark && benchmark.evaluated_at
      ? "Generated " + escapeHtml(new Date(benchmark.evaluated_at).toLocaleString())
      : "Generation time not reported";
    const benchmarkPrompt = benchmark && benchmark.production_prompt_version
      ? "Prompt " + escapeHtml(benchmark.production_prompt_version)
      : "Prompt version not reported";
    elements.evaluation.innerHTML = "<p class=\"mb-1\">" + String(cases) + " evaluation cases</p>" +
      "<span>" + metrics + "</span><br><span>" + String(benchmarkVariants) +
      " grader benchmark variants</span><br><span>" + benchmarkTime +
      "</span><br><span>" + benchmarkPrompt + "</span>";
    renderBenchmark(benchmark);
  }

  function formatDuration(value) {
    return typeof value === "number" && Number.isFinite(value)
      ? value.toFixed(1) + " ms"
      : "Timing not reported";
  }

  function renderTimings(response, durationMs) {
    const timings = response.stage_timings_ms || {};
    Object.keys(elements.timings).forEach(function (name) {
      elements.timings[name].textContent = formatDuration(timings[name]);
    });
    elements.duration.textContent = Number.isFinite(durationMs)
      ? "Total " + formatDuration(durationMs)
      : "Total not reported";
  }

  function renderBenchmark(benchmark) {
    if (!benchmark || !benchmark.prompt_variants) {
      elements.benchmark.innerHTML = "<span class=\"hint\">Benchmark data unavailable.</span>";
      return;
    }
    const rows = [];
    Object.keys(benchmark.prompt_variants).forEach(function (variant) {
      const item = benchmark.prompt_variants[variant];
      const metrics = item.metrics || {};
      const results = Array.isArray(item.results) ? item.results : [];
      const relevant = results.filter(function (result) {
        return result.expected_decision === "yes";
      }).length;
      const irrelevant = results.length - relevant;
      const incorrect = results.filter(function (result) {
        return result.error || result.actual_decision !== result.expected_decision;
      }).length;
      const details = results.map(function (result) {
        const outcome = result.error
          ? "error: " + result.error
          : result.actual_decision === result.expected_decision
          ? "match"
          : "incorrect";
        return "<tr><td>" + escapeHtml(result.case_id) + "</td><td>" +
          escapeHtml(result.expected_decision) + "</td><td>" +
          escapeHtml(result.actual_decision || "not reported") + "</td><td>" +
          escapeHtml(outcome) + "</td></tr>";
      }).join("");
      rows.push("<tr><td>" + escapeHtml(variant) + "</td><td>" +
        String(results.length) + " (" + String(relevant) + "/" + String(irrelevant) + ")" +
        "</td><td>" + String(incorrect) + "</td><td>" +
        escapeHtml(metrics.f1_relevant == null ? "not reported" : metrics.f1_relevant) +
        "</td><td>" + escapeHtml(metrics.false_rejection_rate == null ? "not reported" : metrics.false_rejection_rate) +
        "</td></tr><tr><td colspan=\"5\"><details><summary>Case classifications</summary>" +
        "<table class=\"benchmark-table\"><thead><tr><th>Case</th><th>Expected</th><th>Actual</th><th>Result</th></tr></thead>" +
        "<tbody>" + details + "</tbody></table></details></td></tr>");
    });
    elements.benchmark.innerHTML = "<table class=\"benchmark-table\"><caption class=\"visually-hidden\">Grader benchmark variants</caption>" +
      "<thead><tr><th>Variant</th><th>Cases</th><th>Incorrect/errors</th><th>F1</th><th>False rejection</th></tr></thead>" +
      "<tbody>" + rows.join("") + "</tbody></table>";
  }

  function renderSources(sources) {
    const uniqueSources = [];
    const seen = new Set();
    (Array.isArray(sources) ? sources : []).forEach(function (source) {
      const key = [
        source.kind || "unknown",
        source.source_id || "",
        source.title || "",
        source.url || "",
      ].join("|");
      if (!seen.has(key)) {
        seen.add(key);
        uniqueSources.push(source);
      }

      function renderDocuments(report) {
        const documents = report && Array.isArray(report.items) ? report.items : [];
        if (!documents.length) {
          elements.documents.className = "source-list empty-state compact mt-3";
          elements.documents.innerHTML = "<p class=\"mb-1\">No documents loaded</p><span>Upload a supported document to build the knowledge base.</span>";
          return;
        }
        elements.documents.className = "source-list mt-3";
        elements.documents.innerHTML = documents.map(function (item) {
          return "<div class=\"source-item\"><span class=\"source-index\">" +
            escapeHtml(item.file_format.toUpperCase()) + "</span><div><div class=\"source-id\">" +
            escapeHtml(item.original_filename) + "</div><div class=\"hint\">" +
            escapeHtml(item.status + " · " + item.chunk_count + " chunks") +
            "</div></div><div><button class=\"btn btn-ghost btn-sm document-reindex\" data-document-id=\"" +
            escapeHtml(item.document_id) + "\">Reindex</button> <button class=\"btn btn-ghost btn-sm document-delete\" data-document-id=\"" +
            escapeHtml(item.document_id) + "\">Delete</button></div></div>";
        }).join("");
      }
    });
    elements.sourceCount.textContent = String(uniqueSources.length);
    if (!uniqueSources.length) {
      elements.sources.className = "source-list empty-state compact";
      elements.sources.innerHTML = "<p class=\"mb-1\">No grounded sources</p><span>The API returned no source attribution.</span>";
      return;
    }
    elements.sources.className = "source-list";
    elements.sources.innerHTML = uniqueSources.map(function (source, index) {
      const sourceId = source.source_id || source.title || "Unidentified source";
      const kind = source.kind || "unknown";
      return "<div class=\"source-item\"><span class=\"source-index\">" +
        String(index + 1).padStart(2, "0") +
        "</span><div><div class=\"source-kind\">" +
        escapeHtml(kind) +
        " source</div><div class=\"source-id\">" +
        escapeHtml(sourceId) +
        "</div></div></div>";
    }).join("");
  }

  function renderAnswer(response, durationMs) {
    const hasAnswer = typeof response.answer === "string" && response.answer.trim().length > 0;
    const diagnostics = Array.isArray(response.diagnostics) ? response.diagnostics : [];
    renderTimings(response, durationMs);
    elements.answer.className = "answer-region";
    let badge;
    let heading;
    let text;
    if (hasAnswer) {
      badge = response.fallback_used ? "<span class=\"result-badge fallback\">Fallback answer</span>" : "<span class=\"result-badge grounded\">Grounded answer</span>";
      heading = "<p class=\"eyebrow mb-2\">Response</p>";
      text = escapeHtml(response.answer);
    } else {
      badge = "<span class=\"result-badge failure\">Safe failure</span>";
      heading = "<p class=\"eyebrow mb-2\">No grounded answer</p>";
      text = "CRAG did not produce a grounded answer. No unsupported claim was shown.";
    }
    const diagnosticMarkup = diagnostics.length
      ? "<div class=\"mt-3\"><span class=\"hint\">Diagnostics: " + escapeHtml(diagnostics.join(", ")) + "</span></div>"
      : "";
    elements.answer.innerHTML = "<div class=\"result-meta\">" + badge + "</div>" + heading +
      "<div class=\"answer-content" + (hasAnswer ? "" : " safe-failure") + "\">" +
      text + "</div>" + diagnosticMarkup;
    renderSources(response.sources);
    renderEvidence(response);
    renderWebStatus(response.web_search);
    const stages = response.stage_statuses || {};
    Object.keys(elements.stages).forEach(function (name) {
      setStage(name, stages[name] || "Not reported");
    });
  }

  function renderError(error, durationMs) {
    renderTimings(error.body || {}, durationMs);
    elements.answer.className = "answer-region";
    const diagnostics = error.body && Array.isArray(error.body.diagnostics) ? error.body.diagnostics : [];
    const diagnosticMarkup = diagnostics.length
      ? "<div class=\"mt-3\"><span class=\"hint\">Diagnostics: " + escapeHtml(diagnostics.join(", ")) + "</span></div>"
      : "";
    elements.answer.innerHTML = "<div class=\"result-meta\"><span class=\"result-badge failure\">HTTP " +
      String(error.status || "error") + " · Safe failure</span></div><p class=\"eyebrow mb-2\">Request unavailable</p>" +
      "<div class=\"answer-content safe-failure\">" + escapeHtml(error.message) + "</div>" + diagnosticMarkup;
    renderSources(error.body && error.body.sources);
    renderEvidence(error.body || {});
    renderWebStatus(error.body && error.body.web_search);
    const stages = error.body && error.body.stage_statuses ? error.body.stage_statuses : {};
    Object.keys(elements.stages).forEach(function (name) {
      setStage(name, stages[name] || "Not reported");
    });
  }

  function renderHealth(kind, healthy, label) {
    const dot = kind === "live" ? elements.liveDot : elements.readyDot;
    const status = kind === "live" ? elements.liveStatus : elements.readyStatus;
    dot.classList.toggle("ok", healthy);
    dot.classList.toggle("fail", !healthy);
    status.textContent = label;
  }

  function setStatusUpdated(text) {
    elements.statusUpdated.textContent = text;
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, function (character) {
      return {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#039;"}[character];
    });
  }

  window.CragUi = {
    clearAlert: clearAlert,
    elements: elements,
    renderAnswer: renderAnswer,
    renderError: renderError,
    renderHealth: renderHealth,
    renderDocuments: renderDocuments,
    renderEvaluation: renderEvaluation,
    renderTimings: renderTimings,
    setLoading: setLoading,
    setStage: setStage,
    setStatusUpdated: setStatusUpdated,
    showAlert: showAlert,
  };
}());
