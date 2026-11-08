(function () {
  "use strict";

  const questionInput = document.getElementById("question");
  const form = document.getElementById("ask-form");
  const clearButton = document.getElementById("clear-question");
  const refreshButtons = [
    document.getElementById("refresh-status"),
    document.getElementById("refresh-status-inline"),
  ];

  async function refreshStatus() {
    CragUi.renderHealth("live", false, "Checking");
    CragUi.renderHealth("ready", false, "Checking");
    CragUi.setStatusUpdated("Checking the API...");
    const results = await Promise.allSettled([
      CragApi.getHealth("/health/live"),
      CragApi.getHealth("/health/ready"),
    ]);
    const live = results[0];
    const ready = results[1];
    CragUi.renderHealth("live", live.status === "fulfilled" && live.value.status === "healthy", live.status === "fulfilled" ? live.value.status : "Unavailable");
    CragUi.renderHealth("ready", ready.status === "fulfilled" && ready.value.status === "ready", ready.status === "fulfilled" ? ready.value.status : "Unavailable");
    CragUi.setStatusUpdated("Last checked " + new Date().toLocaleTimeString());
  }

  async function refreshEvaluation() {
    const results = await Promise.allSettled([
      CragApi.getEvaluation("/api/v1/evaluation"),
      CragApi.getEvaluation("/api/v1/evaluation/benchmark"),
    ]);
    CragUi.renderEvaluation(
      results[0].status === "fulfilled" ? results[0].value : null,
      results[1].status === "fulfilled" ? results[1].value : null
    );
  }

  async function refreshDocuments() {
    try {
      CragUi.renderDocuments(await CragApi.getDocuments({limit: "50"}));
    } catch (error) {
      CragUi.renderDocuments(null);
    }
  }

  async function uploadDocuments() {
    const input = document.getElementById("document-files");
    if (!input.files.length) return;
    try {
      await CragApi.uploadDocuments(Array.from(input.files));
      input.value = "";
      await refreshDocuments();
    } catch (error) {
      CragUi.showAlert(error.message);
    }
  }

  async function submitQuestion(event) {
    event.preventDefault();
    CragUi.clearAlert();
    const question = questionInput.value.trim();
    if (!question) {
      questionInput.focus();
      CragUi.showAlert("Enter a question before running retrieval.");
      return;
    }
    CragUi.setLoading(true);
    const started = performance.now();
    try {
      const response = await CragApi.ask(question);
      CragUi.renderAnswer(response, performance.now() - started);
    } catch (error) {
      CragUi.renderError(error, performance.now() - started);
    } finally {
      CragUi.setLoading(false);
    }
  }

  questionInput.addEventListener("input", function () {
    CragUi.elements.count.textContent = questionInput.value.length + " / 10,000";
  });
  questionInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", submitQuestion);
  clearButton.addEventListener("click", function () {
    questionInput.value = "";
    CragUi.elements.count.textContent = "0 / 10,000";
    CragUi.clearAlert();
    questionInput.focus();
  });
  refreshButtons.forEach(function (button) {
    button.addEventListener("click", refreshStatus);
  });
  document.getElementById("refresh-documents").addEventListener("click", refreshDocuments);
  document.getElementById("upload-documents").addEventListener("click", uploadDocuments);
  document.getElementById("documents-region").addEventListener("click", async function (event) {
    const reindex = event.target.closest(".document-reindex");
    if (reindex) {
      try {
        await CragApi.reindexDocument(reindex.dataset.documentId);
        await refreshDocuments();
      } catch (error) {
        CragUi.showAlert(error.message);
      }
      return;
    }
    const button = event.target.closest(".document-delete");
    if (button && window.confirm("Delete this document and its indexed chunks?")) {
      try {
        await CragApi.deleteDocument(button.dataset.documentId);
        await refreshDocuments();
      } catch (error) {
        CragUi.showAlert(error.message);
      }
    }
  });
  refreshStatus();
  refreshEvaluation();
  refreshDocuments();
}());
