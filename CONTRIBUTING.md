# MindSpore Golden Stick Contributing Guide

[查看中文](./CONTRIBUTING_CN.md)

<!-- TOC -->

- [MindSpore Golden Stick Contributing Guide](#mindspore-golden-stick-contributing-guide)
    - [Contributor License Agreement](#contributor-license-agreement)
    - [Getting Started](#getting-started)
    - [Contribution Workflow](#contribution-workflow)
        - [Code Style](#code-style)
        - [Local Code Self-Check](#local-code-self-check)
        - [Fork-Pull Development Model](#fork-pull-development-model)
        - [Reporting Issues](#reporting-issues)
        - [Propose PRs](#propose-prs)

<!-- /TOC -->

## Contributor License Agreement

It's required to sign CLA before your first code submission to MindSpore community.

For individual contributor, please refer to [ICLA online document](https://www.mindspore.cn/icla) for the detailed information.

## Getting Started

- Fork the repository on [Gitee](https://gitee.com/mindspore/golden-stick).
- Refer to [README](./README.md) and [Installation Guide](./docs/en/install.md) for project information and build instructions.
- Refer to [Quick Start](./example) to try applying SmoothQuant to the Qwen3-0.6B network.
- Refer to [Architecture Design](./docs/en/design.md) to understand the project architecture, and refer to [wiki](https://gitee.com/mindspore/golden-stick/wikis) for detailed software design of the project.

## Contribution Workflow

### Code Style

Please follow this style to make MindSpore Golden Stick community easy to review, maintain and develop.

- Coding Guidelines

    The MindSpore Golden Stick community uses [Python PEP 8 Coding Style](https://pep8.org/). It is recommended to install the following plugins in your IDE for code format checking: [CodeSpell](https://github.com/codespell-project/codespell), [Lizard](http://www.lizard.ws), [ShellCheck](https://github.com/koalaman/shellcheck), and [PyLint](https://pylint.org).

- Unittest Guidelines

    The MindSpore Golden Stick community uses Python unit testing framework [pytest](http://www.pytest.org/en/latest/). Comment names should reflect the design intent of the test cases.

- Refactoring Guidelines

    We encourage developers to refactor our code to eliminate [code smells](https://en.wikipedia.org/wiki/Code_smell). All code should conform to coding style and testing style, and refactored code is no exception. The [Lizard](http://www.lizard.ws) threshold for non-commented lines of code (nloc) is 100, and the threshold for cyclomatic complexity (cnc) is 20. When receiving Lizard warnings, the code to be merged must be refactored.

- Documentation Guidelines

    We use *MarkdownLint* to check Markdown document format. MindSpore CI has modified the following rules based on the default configuration:
    - MD007 (Unordered list indentation): The parameter **indent** is set to **4**, indicating that all content in unordered lists needs to be indented by 4 spaces.
    - MD009 (Trailing spaces): The parameter **br_spaces** is set to **2**, indicating that there can be 0 or 2 spaces at the end of a line.
    - MD029 (Ordered list item prefix): The parameter **style** is set to **ordered**, indicating ascending order.

    For details, please refer to [Rules](https://github.com/markdownlint/markdownlint/blob/master/docs/RULES.md).

### Local Code Self-Check

During development, it is recommended to use the pre-push feature for local code self-check, which can perform code scanning similar to the Code Check phase in CI gates locally, improving the success rate when running gates during code submission. For usage instructions, please refer to [pre-push Quick Guide](./scripts/pre_push/README.md).

### Fork-Pull Development Model

- Fork MindSpore Golden Stick Repository

    Before submitting code to the MindSpore Golden Stick project, please ensure that this project has been forked to your own repository. There may be parallel development between the MindSpore Golden Stick repository and your own repository, so please pay attention to consistency between them.

- Clone Remote Repository

    If you want to download the code to your local computer, it's best to use the git method:

    ```shell
    # On Gitee:
    git clone https://gitee.com/{insert_your_forked_repo}/golden-stick.git
    git remote add upstream https://gitee.com/mindspore/golden-stick.git
    ```

- Develop Code Locally

    To avoid branch inconsistency, it is recommended to switch to a new branch:

    ```shell
    git checkout -b {new_branch_name} origin/master
    ```

    Taking the master branch as an example, if MindSpore Golden Stick needs to create version branches and downstream development branches, please fix upstream bugs first, then change the code.

- Push Code to Remote Repository

    After updating the code, push the updates in a formal way:

    ```shell
    git add .
    git status # Check update status
    git commit -m "Your commit title"
    git commit -s --amend # Add specific description of the commit
    git pull upstream master --rebase # Merge upstream changes
    git push origin {new_branch_name}
    ```

- Pull Request to MindSpore Golden Stick Repository

    In the final step, you need to pull a comparison request between the new branch and the MindSpore Golden Stick main branch. After completing the pull request, Jenkins CI will be automatically set up for build testing. Pull requests should be merged into the upstream master branch as soon as possible to reduce merge risks.

### Reporting Issues

After discovering problems, it is recommended to contribute to the MindSpore Golden Stick project by reporting issues. Bug reports should be written as standardized as possible with detailed content. Thank you for your contribution to the project.

When reporting issues, please refer to the following format:

- Describe the environment versions you are using (MindSpore Golden Stick, OS, Python, etc.).
- Specify whether it's a bug report or feature request.
- Specify the issue type, adding labels can highlight the issue on the issue board.
- What is the problem?
- How do you expect it to be handled?
- How to reproduce? (Describe as precisely and specifically as possible)
- Special notes for reviewers.

**Issue Consultation:**

- **When resolving an issue, please comment first** to inform others that you are responsible for resolving the issue.
- **For issues that have been open for a long time**, it is recommended that contributors perform a pre-check before resolving the issue.
- **If you resolve an issue you reported yourself**, you still need to inform others before closing the issue.
- **For quick issue response**, you can add labels to issues. For label details, refer to [Label List](https://gitee.com/mindspore/community/blob/master/sigs/dx/docs/labels.md).

### Proposing PRs

- Propose your ideas as an *issue* on [Gitee](https://gitee.com/mindspore/golden-stick/issues).
- If it's a new feature that requires extensive design details, you should also submit a design proposal. The design proposal also needs to be reviewed in [MindSpore LLM Inference Serving SIG](https://www.mindspore.cn/sig/LLM%20Inference%20Serving).
- After reaching consensus through MindSpore Golden Stick community issue discussions and design proposal reviews, develop in the forked repository and submit a PR.
- After sufficient discussion, merge, abandon, or reject the PR based on the discussion results.

**PRs Consultation:**

- Avoid unrelated changes.
- Ensure your commit history is organized.
- Ensure your branch is always consistent with the main branch.
- In PRs for bug fixes, ensure all related issues are linked.
