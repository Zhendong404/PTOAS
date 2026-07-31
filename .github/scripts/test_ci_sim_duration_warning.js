// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const observeRun = require('./ci_sim_duration_warning.js');

async function main() {
  const contextDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ci-sim-duration-'));
  const sha = 'a'.repeat(40);
  fs.writeFileSync(path.join(contextDir, 'pr-number'), '42\n');
  fs.writeFileSync(path.join(contextDir, 'pr-head-sha'), `${sha}\n`);
  const jobs = [
    {name: 'build-wheel-x86_64 / Build ptoas wheel (x86_64, py3.11)', started_at: '2026-01-01T00:00:00Z', completed_at: '2026-01-01T00:11:00Z', conclusion: 'success'},
    {name: 'vpto-sim', started_at: '2026-01-01T00:11:00Z', completed_at: '2026-01-01T00:20:00Z', conclusion: 'success'},
    {name: 'ci-sim-required', started_at: '2026-01-01T01:40:00Z', completed_at: '2026-01-01T01:41:00Z', conclusion: 'success'},
  ];
  const github = {
    rest: {
      pulls: {get: async () => ({data: {state: 'open', head: {sha}, labels: [], user: {login: 'author'}}})},
      actions: {listJobsForWorkflowRun: async () => ({data: {jobs}})},
      issues: {listComments: async () => ({data: []})},
    },
    paginate: async (method, args) => (await method(args)).data.jobs || (await method(args)).data,
  };
  const result = await observeRun({
    github,
    context: {repo: {owner: 'owner', repo: 'repo'}, payload: {workflow_run: {id: 1, head_sha: sha, html_url: 'https://example.invalid/run'}}},
    config: {contextDir, dryRun: true, softTimeoutMinutes: 90},
  });
  assert.match(result, /P95 sampling budget: \*\*10m\*\*/);
  assert.match(result, /Producer → gate critical path: \*\*1h 41m 0s\*\*/);
  assert.match(result, /`vpto-sim`: \*\*9m 0s\*\*/);
  fs.rmSync(contextDir, {recursive: true});
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
