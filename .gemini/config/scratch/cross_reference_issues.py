import json
import re
import subprocess
from pathlib import Path

issues_file = r"C:\Users\User\.gemini\antigravity-ide\brain\fcfe5e66-19ad-439e-befa-60e9234dd738\github_issues.json"
git_exe = r"C:\Program Files\Git\cmd\git.exe"
repo_dir = r"c:\Users\User\Documents\GitHub\TradingAgents"
output_file = r"C:\Users\User\.gemini\antigravity-ide\brain\fcfe5e66-19ad-439e-befa-60e9234dd738\issue_resolution_report.md"

# Define keyword mapping for issues to verify if they are addressed via specific code commits
issue_verification_keywords = {
    1100: ["AKShare", "akshare", "973d133"],
    997: ["smoke", "structured output markdown", "cbd17ac"],
    920: ["default configuration", "default config", "0585cb1"],
    807: ["fsspec", "save_report_to_disk", "758fc55"],
    805: ["model cutoff", "Knowledge Leakage", "940", "c8e259b"],
    784: ["flicker", "784", "Castle UI", "8448c68"],
    752: ["run_config", "752"],
    750: ["prompt caching", "878", "ab42f1f"],
    605: ["Ollama", "686", "7905158"],
    561: ["earnings-context", "835", "earnings calendar", "5f8af1a"],
    545: ["configurable metrics", "581"],
    542: ["pre-computed indicator", "584", "361b933"],
    524: ["spend limits", "582", "524"],
    514: ["pre-trade market", "523", "514"],
    433: ["execution planning", "487", "Parallelize", "939e532"],
    406: ["multi-stock comparison", "456", "portfolio analysis", "3f1886a"],
    291: ["max_summary_chars", "386"],
    275: ["multi-provider", "324", "275", "276", "705886f"],
    274: ["get_global_news_openai", "277", "db6a107"],
    264: ["generic AI trading agent", "593", "264", "ee7fda8"]
}

def get_local_commit_messages():
    cmd = [
        git_exe,
        "-C", repo_dir,
        "log",
        "--format=%H|%s|%b"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        raise Exception(f"Failed to get git log: {result.stderr}")
    return result.stdout.strip().split('\n')

def main():
    with open(issues_file, 'r', encoding='utf-8') as f:
        items = json.load(f)
        
    issues = [item for item in items if not item.get("is_pr")]
    prs = [item for item in items if item.get("is_pr")]
    
    issues.sort(key=lambda x: x["number"], reverse=True)
    open_issues = [i for i in issues if i["state"] == "open"]
    
    commits = get_local_commit_messages()
    local_commit_info = []
    for c in commits:
        if not c:
            continue
        parts = c.split('|', 2)
        h = parts[0]
        subj = parts[1] if len(parts) > 1 else ""
        body = parts[2] if len(parts) > 2 else ""
        local_commit_info.append((h, subj, body))
        
    # Map PRs to issue numbers they reference
    pr_to_issues = {}
    for pr in prs:
        text = f"{pr['title']} {pr['body']}"
        ref_issues = re.findall(r'#(\d+)\b', text)
        ref_issues_named = re.findall(r'(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\s+#?(\d+)\b', text, re.IGNORECASE)
        all_refs = set(int(x) for x in (ref_issues + ref_issues_named))
        if all_refs:
            pr_to_issues[pr['number']] = all_refs

    report_entries = []
    for issue in open_issues:
        iss_num = issue['number']
        title = issue['title']
        created_at = issue['created_at'][:10]
        html_url = issue['html_url']
        
        referencing_prs = []
        for pr_num, referenced in pr_to_issues.items():
            if iss_num in referenced:
                pr_item = next((p for p in prs if p['number'] == pr_num), None)
                if pr_item:
                    referencing_prs.append(pr_item)
                    
        addressed_status = "No proposed fix found"
        fix_details = []
        
        # Comprehensive search for local addressal
        is_locally_addressed = False
        matching_commits = []
        
        # 1. Search commit log directly for "#issue_num"
        for h, subj, body in local_commit_info:
            if f"#{iss_num}" in f"{subj} {body}":
                is_locally_addressed = True
                matching_commits.append(f"Commit `{h[:7]}`: {subj}")
                
        # 2. Search commit log for associated PRs
        for pr in referencing_prs:
            pr_n = pr['number']
            # Search by PR number in log
            for h, subj, body in local_commit_info:
                # regex to check for pr number
                if re.search(r'(?:PR|pr|pull request)\s*[\/#]?\s*' + str(pr_n) + r'\b', f"{subj} {body}", re.IGNORECASE):
                    is_locally_addressed = True
                    matching_commits.append(f"Commit `{h[:7]}` (PR #{pr_n}): {subj}")
                    
        # 3. Search commit log using custom keywords/hashes
        if iss_num in issue_verification_keywords:
            keywords = issue_verification_keywords[iss_num]
            for kw in keywords:
                for h, subj, body in local_commit_info:
                    if kw.lower() in f"{h} {subj} {body}".lower():
                        is_locally_addressed = True
                        matching_commits.append(f"Commit `{h[:7]}` (keyword '{kw}'): {subj}")

        # Deduplicate commits
        matching_commits = list(set(matching_commits))

        if is_locally_addressed:
            addressed_status = "Addressed (Fix integrated locally)"
            fix_details.extend(matching_commits)
        elif referencing_prs:
            addressed_status = "Proposed fix PR found, but not merged locally"
            for pr in referencing_prs:
                fix_details.append(f"PR #{pr['number']} ({pr['state']} upstream): {pr['title']}")
                
        report_entries.append({
            "number": iss_num,
            "title": title,
            "created_at": created_at,
            "url": html_url,
            "status": addressed_status,
            "details": fix_details
        })

    # Sort report entries by status then issue number descending
    report_entries.sort(key=lambda x: (x['status'], -x['number']))

    md = []
    md.append("# Upstream Open Issues Resolution Report")
    md.append("")
    md.append(f"Analyzing **{len(open_issues)}** currently open issues from the upstream repository `TauricResearch/TradingAgents` and checking if they have been addressed in our local codebase `rajatvarna/TradingAgents`.")
    md.append("")
    
    status_counts = {}
    for entry in report_entries:
        status_counts[entry['status']] = status_counts.get(entry['status'], 0) + 1
        
    md.append("## Summary of Addressal Status")
    md.append("")
    for status, count in status_counts.items():
        md.append(f"- **{status}**: {count}")
    md.append("")
    
    md.append("## Detailed Issue Status")
    md.append("")
    md.append("| Issue | Title | Status | Fix / PR Details |")
    md.append("| :--- | :--- | :--- | :--- |")
    for entry in report_entries:
        details_str = "<br>".join(entry['details']) if entry['details'] else "-"
        # Escape any markdown in details to prevent table breakage
        details_str = details_str.replace("|", "\\|")
        md.append(f"| [#{entry['number']}]({entry['url']}) | {entry['title']} | **{entry['status']}** | {details_str} |")
        
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
        
    print(f"Report generated successfully: {output_file}")

if __name__ == '__main__':
    main()
