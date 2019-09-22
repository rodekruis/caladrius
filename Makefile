image_name := "gulfaraz/caladrius"
version_production := `cat VERSION`
version_git := $$(git log -1 --format=%h)

build_production:
	docker build --no-cache --pull -t ${image_name}:${version_production} .

build_fast: 
	docker build -t ${image_name}:${version_git} .
