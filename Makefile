.PHONY: status check

status:
	git status --short --branch

check:
	@echo "No project-specific check target configured."
