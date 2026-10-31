const fs = require("fs");
const path = require("path");

module.exports = function handler(_request, response) {
  const page = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");
  response.statusCode = 200;
  response.setHeader("Content-Type", "text/html; charset=utf-8");
  response.setHeader("Cache-Control", "no-store, max-age=0");
  response.end(page);
};
