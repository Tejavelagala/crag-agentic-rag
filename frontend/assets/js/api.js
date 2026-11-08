(function () {
  "use strict";

  async function parseResponse(response) {
    let body = null;
    try {
      body = await response.json();
    } catch (error) {
      body = { detail: "The API returned an unreadable response." };
    }
    if (!response.ok) {
      const requestError = new Error(
        body.detail || "The API could not complete the request."
      );
      requestError.status = response.status;
      requestError.body = body;
      throw requestError;
    }
    return body;
  }

  async function getHealth(path) {
    const response = await fetch(path, { headers: { Accept: "application/json" } });
    return parseResponse(response);
  }

  async function ask(question) {
    const response = await fetch("/api/v1/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ question: question }),
    });
    return parseResponse(response);
  }

  async function getEvaluation(path) {
    const response = await fetch(path, { headers: { Accept: "application/json" } });
    return parseResponse(response);
  }

  async function getDocuments(query) {
    const params = new URLSearchParams(query || {});
    const response = await fetch("/api/v1/documents?" + params.toString(), {
      headers: { Accept: "application/json" },
    });
    return parseResponse(response);
  }

  async function uploadDocuments(files) {
    const form = new FormData();
    files.forEach(function (file) { form.append("files", file); });
    const response = await fetch("/api/v1/documents", { method: "POST", body: form });
    return parseResponse(response);
  }

  async function deleteDocument(documentId) {
    const response = await fetch("/api/v1/documents/" + encodeURIComponent(documentId), {
      method: "DELETE",
    });
    return parseResponse(response);
  }

  async function reindexDocument(documentId) {
    const response = await fetch("/api/v1/documents/" + encodeURIComponent(documentId) + "/reindex", {
      method: "POST",
    });
    return parseResponse(response);
  }

  window.CragApi = {
    ask: ask,
    getEvaluation: getEvaluation,
    getDocuments: getDocuments,
    getHealth: getHealth,
    uploadDocuments: uploadDocuments,
    deleteDocument: deleteDocument,
    reindexDocument: reindexDocument,
  };
}());
