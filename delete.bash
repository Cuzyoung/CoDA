git filter-branch --force --index-filter \
	  "git rm --cached --ignore-unmatch pretrained/mit_b5.pth" \
	    --prune-empty --tag-name-filter cat -- --all

