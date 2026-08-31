pluginManagement {
    repositories {
        google()
        // 显式使用 mavenCentral 镜像地址（repo1.maven.org），规避默认的 repo.maven.apache.org
        maven { url = uri("https://repo1.maven.org/maven2/") }
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        maven { url = uri("https://repo1.maven.org/maven2/") }
    }
}

rootProject.name = "AndroidYOLO"
include(":app")