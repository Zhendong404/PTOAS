// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

const fs = require('fs');
const path = require('path');

const COMMENT_MARKER = '<!-- ci-sim-duration-warning -->';
const RESOLVED_PREFIX = 'Resolved:';
const BOT_LOGIN = 'github-actions[bot]';
const PRODUCER_JOB_NAME = 'build-and-test-wheel';
const GATE_JOB_NAME = 'ci-sim-required';
const SUITE_JOB_NAMES = ['pypto-sim-smoke', 'tileop-st', 'vpto-sim', 'tilelib-st', 'ptodsl-st'];
const PRODUCER_TARGET_MINUTES = 6;
const PRODUCER_P95_BUDGET_MINUTES = 10;
const SLOW_LABEL = 'ci-slow';
const SHA_RE = /^[0-9a-fA-F]{40}$/;

function readPrContext(contextDir) {
  const numberPath = path.join(contextDir, 'pr-number');
  const shaPath = path.join(contextDir, 'pr-head-sha');
  if (!fs.existsSync(numberPath) || !fs.existsSync(shaPath)) {
    return null;
  }

  const numberText = fs.readFileSync(numberPath, 'utf8').trim();
  const headSha = fs.readFileSync(shaPath, 'utf8').trim();
  const number = Number(numberText);
  if (!/^\d+$/.test(numberText) || !Number.isSafeInteger(number) || number <= 0 || !SHA_RE.test(headSha)) {
    throw new Error(`Invalid PR context in ${contextDir}`);
  }
  return {number, headSha};
}

function formatDuration(totalSeconds) {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return hours ? `${hours}h ${minutes}m ${seconds}s` : `${minutes}m ${seconds}s`;
}

function formatBudget(totalMinutes) {
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (hours && minutes) return `${hours}h ${minutes}m`;
  if (hours) return `${hours}h`;
  return `${minutes}m`;
}

function warningBody(author, producerElapsed, criticalPath, budget, conclusion, runUrl, suiteDurations) {
  const suiteLines = suiteDurations.length
    ? suiteDurations.map(item => `- \`${item.name}\`: **${item.duration}**`).join('\n')
    : '- No completed suite jobs were found.';
  return [
    COMMENT_MARKER,
    `Warning: @${author}, ci-sim exceeded its **${budget} critical-path soft budget**.`,
    '',
    `- \`${PRODUCER_JOB_NAME}\` runtime: **${producerElapsed}**`,
    `- Producer → gate critical path: **${criticalPath}**`,
    `- Selected suite durations:\n${suiteLines}`,
    `- Warm-cache producer sampling target: **${PRODUCER_TARGET_MINUTES}m**; P95 sampling budget: **${PRODUCER_P95_BUDGET_MINUTES}m**`,
    '- Producer sampling targets do not trigger this per-run warning because a cold LLVM cache is expected after an LLVM revision change.',
    `- Job conclusion: **${conclusion}**`,
    `- [Workflow run](${runUrl})`,
    '',
    'This warning is advisory only and does not affect required checks. ' +
      'Please inspect the step timings for an unexpected regression.',
  ].join('\n');
}

function resolvedBody(producerElapsed, criticalPath, budget, runUrl) {
  return [
    COMMENT_MARKER,
    'Resolved: ci-sim runtime is back within its soft budget.',
    '',
    `- Latest \`${PRODUCER_JOB_NAME}\` runtime: **${producerElapsed}**`,
    `- Producer → gate critical path: **${criticalPath}**`,
    `- Warm-cache producer sampling target: **${PRODUCER_TARGET_MINUTES}m**; P95 sampling budget: **${PRODUCER_P95_BUDGET_MINUTES}m**`,
    `- Critical-path soft budget: **${budget}**`,
    `- [Workflow run](${runUrl})`,
    '',
    'The previous duration warning is resolved. This status is advisory only.',
  ].join('\n');
}

function positiveInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

module.exports = async function observeRun({github, context, config = {}}) {
  const owner = config.owner || context.repo.owner;
  const repo = config.repo || context.repo.repo;
  const workflowRun = context.payload.workflow_run || {};
  const runId = positiveInteger(config.runId || workflowRun.id, 'run ID');
  const runHeadSha = config.runHeadSha || workflowRun.head_sha;
  const runUrl = config.runUrl || workflowRun.html_url;
  const softTimeoutMinutes = positiveInteger(
    config.softTimeoutMinutes || process.env.SOFT_TIMEOUT_MINUTES || 90,
    'soft timeout',
  );
  const contextDir = config.contextDir || process.env.PR_CONTEXT_DIR || 'pr-context';
  const dryRun = config.dryRun === true;

  if (!SHA_RE.test(runHeadSha || '')) {
    throw new Error(`Invalid run head SHA: ${runHeadSha}`);
  }

  const prContext = readPrContext(contextDir);
  if (!prContext) {
    return `No PR context found for workflow run ${runId}; skipping duration warning.`;
  }

  const pullResponse = await github.rest.pulls.get({owner, repo, pull_number: prContext.number});
  const pull = pullResponse.data;
  if (pull.state !== 'open') {
    return `PR #${prContext.number} is ${pull.state}; skipping duration warning.`;
  }
  if (prContext.headSha !== runHeadSha || pull.head.sha !== runHeadSha) {
    return `Workflow run ${runId} is stale for PR #${prContext.number}; skipping duration warning.`;
  }
  const hasSlowLabel = pull.labels.some(label => label.name === SLOW_LABEL);

  const jobs = await github.paginate(github.rest.actions.listJobsForWorkflowRun, {
    owner,
    repo,
    run_id: runId,
    filter: 'latest',
    per_page: 100,
  });
  const producer = jobs.find(item => item.name.includes(PRODUCER_JOB_NAME) && item.started_at && item.completed_at);
  const gate = jobs.find(item => item.name === GATE_JOB_NAME && item.started_at && item.completed_at);
  if (!producer) {
    return `No completed ${PRODUCER_JOB_NAME} job found in workflow run ${runId}.`;
  }

  const elapsedSeconds = Math.floor((Date.parse(producer.completed_at) - Date.parse(producer.started_at)) / 1000);
  if (!Number.isFinite(elapsedSeconds) || elapsedSeconds < 0) {
    throw new Error(`Invalid job timestamps: ${producer.started_at} to ${producer.completed_at}`);
  }

  const elapsed = formatDuration(elapsedSeconds);
  const criticalPathSeconds = gate
    ? Math.floor((Date.parse(gate.completed_at) - Date.parse(producer.started_at)) / 1000)
    : elapsedSeconds;
  if (!Number.isFinite(criticalPathSeconds) || criticalPathSeconds < 0) {
    throw new Error(`Invalid producer/gate timestamps: ${producer.started_at} to ${gate?.completed_at}`);
  }
  const criticalPath = formatDuration(criticalPathSeconds);
  const suiteDurations = jobs
    .filter(item => SUITE_JOB_NAMES.includes(item.name) && item.started_at && item.completed_at)
    .map(item => ({
      name: item.name,
      duration: formatDuration(Math.max(0, Math.floor((Date.parse(item.completed_at) - Date.parse(item.started_at)) / 1000))),
    }));
  const budget = formatBudget(softTimeoutMinutes);
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: prContext.number,
    per_page: 100,
  });
  const existing = comments
    .slice()
    .reverse()
    .find(comment => comment.user?.login === BOT_LOGIN && comment.body?.includes(COMMENT_MARKER));

  const criticalPathOverBudget = criticalPathSeconds > softTimeoutMinutes * 60;
  if (criticalPathOverBudget) {
    const body = warningBody(pull.user.login, elapsed, criticalPath, budget, producer.conclusion || 'unknown', runUrl, suiteDurations);
    if (dryRun) {
      const labelAction = hasSlowLabel ? 'keep' : 'add';
      const commentAction = existing ? 'update' : 'create';
    return `Dry run: would ${labelAction} ${SLOW_LABEL} and ${commentAction} slow CI warning ` +
        `on PR #${prContext.number}.\n${body}`;
    }

    if (!hasSlowLabel) {
      await github.rest.issues.addLabels({owner, repo, issue_number: prContext.number, labels: [SLOW_LABEL]});
    }
    if (existing) {
      await github.rest.issues.updateComment({owner, repo, comment_id: existing.id, body});
    } else {
      await github.rest.issues.createComment({owner, repo, issue_number: prContext.number, body});
    }
    const labelAction = hasSlowLabel ? 'Kept' : 'Applied';
    const commentAction = existing ? 'updated' : 'created';
    return `${labelAction} ${SLOW_LABEL}; ${commentAction} slow CI warning on PR #${prContext.number}.`;
  }

  const commentNeedsResolution = Boolean(existing && !existing.body?.includes(RESOLVED_PREFIX));
  if (dryRun && commentNeedsResolution) {
    const actions = [];
    if (hasSlowLabel) actions.push(`remove ${SLOW_LABEL}`);
    if (commentNeedsResolution) actions.push('resolve slow CI warning');
    return `Dry run: would ${actions.join(' and ')} on PR #${prContext.number}.\n` +
      resolvedBody(elapsed, criticalPath, budget, runUrl);
  }

  const actions = [];
  if (hasSlowLabel && commentNeedsResolution) {
    await github.rest.issues.removeLabel({owner, repo, issue_number: prContext.number, name: SLOW_LABEL});
    actions.push(`Removed ${SLOW_LABEL}`);
  }
  if (commentNeedsResolution) {
    await github.rest.issues.updateComment({
      owner,
      repo,
      comment_id: existing.id,
      body: resolvedBody(elapsed, criticalPath, budget, runUrl),
    });
    actions.push('resolved slow CI warning');
  }
  if (actions.length) {
    return `${actions.join('; ')} on PR #${prContext.number}.`;
  }
  return `CI producer runtime ${elapsed}; producer-to-gate critical path ${criticalPath}; within the ${budget} budget.`;
};
