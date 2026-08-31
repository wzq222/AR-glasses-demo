package com.ar.glass.ui;

import org.junit.Test;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.xml.parsers.DocumentBuilderFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;

public class LauncherContractTest {
    private static final String ANDROID_NAMESPACE =
            "http://schemas.android.com/apk/res/android";
    private static final String ACTION_MAIN = "android.intent.action.MAIN";
    private static final String CATEGORY_LAUNCHER = "android.intent.category.LAUNCHER";

    @Test
    public void launchesOnlyThePortraitPhoneInspectionActivity() throws Exception {
        Document manifest = parse(projectFile("app/src/main/AndroidManifest.xml"));
        Element mainActivity = findActivity(manifest, ".ui.MainActivity");
        Element liveActivity = findActivity(manifest, ".ui.LiveInspectionActivity");

        assertNotNull(mainActivity);
        assertNotNull(liveActivity);
        assertFalse(hasLauncherIntent(mainActivity));
        assertEquals(
                Collections.singletonList(".ui.LiveInspectionActivity"),
                launcherActivityNames(manifest));
        assertEquals("true", liveActivity.getAttributeNS(ANDROID_NAMESPACE, "exported"));
        assertEquals("portrait",
                liveActivity.getAttributeNS(ANDROID_NAMESPACE, "screenOrientation"));
    }

    @Test
    public void namesTheInstalledAppAsAPhoneVisionTest() throws Exception {
        Document manifest = parse(projectFile("app/src/main/AndroidManifest.xml"));
        Element application = (Element) manifest.getElementsByTagName("application").item(0);
        assertEquals("@string/app_name",
                application.getAttributeNS(ANDROID_NAMESPACE, "label"));

        Document strings = parse(projectFile("app/src/main/res/values/strings.xml"));
        assertEquals("中车视觉手机测试", stringValue(strings, "app_name"));
    }

    private static List<String> launcherActivityNames(Document manifest) {
        List<String> names = new ArrayList<>();
        NodeList activities = manifest.getElementsByTagName("activity");
        for (int index = 0; index < activities.getLength(); index++) {
            Element activity = (Element) activities.item(index);
            if (hasLauncherIntent(activity)) {
                names.add(activity.getAttributeNS(ANDROID_NAMESPACE, "name"));
            }
        }
        return names;
    }

    private static Element findActivity(Document manifest, String name) {
        NodeList activities = manifest.getElementsByTagName("activity");
        for (int index = 0; index < activities.getLength(); index++) {
            Element activity = (Element) activities.item(index);
            if (name.equals(activity.getAttributeNS(ANDROID_NAMESPACE, "name"))) {
                return activity;
            }
        }
        return null;
    }

    private static boolean hasLauncherIntent(Element activity) {
        NodeList filters = activity.getElementsByTagName("intent-filter");
        for (int filterIndex = 0; filterIndex < filters.getLength(); filterIndex++) {
            Element filter = (Element) filters.item(filterIndex);
            boolean hasMain = false;
            boolean hasLauncher = false;
            NodeList children = filter.getChildNodes();
            for (int childIndex = 0; childIndex < children.getLength(); childIndex++) {
                Node child = children.item(childIndex);
                if (!(child instanceof Element)) {
                    continue;
                }
                Element element = (Element) child;
                String name = element.getAttributeNS(ANDROID_NAMESPACE, "name");
                hasMain |= "action".equals(element.getTagName()) && ACTION_MAIN.equals(name);
                hasLauncher |= "category".equals(element.getTagName())
                        && CATEGORY_LAUNCHER.equals(name);
            }
            if (hasMain && hasLauncher) {
                return true;
            }
        }
        return false;
    }

    private static String stringValue(Document strings, String resourceName) {
        NodeList entries = strings.getElementsByTagName("string");
        for (int index = 0; index < entries.getLength(); index++) {
            Element entry = (Element) entries.item(index);
            if (resourceName.equals(entry.getAttribute("name"))) {
                return entry.getTextContent();
            }
        }
        return null;
    }

    private static Document parse(Path path) throws Exception {
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setNamespaceAware(true);
        return factory.newDocumentBuilder().parse(path.toFile());
    }

    private static Path projectFile(String relativePath) {
        Path projectRoot = Paths.get(System.getProperty("user.dir"));
        Path path = projectRoot.resolve(relativePath);
        if (Files.exists(path)) {
            return path;
        }
        return projectRoot.getParent().resolve(relativePath);
    }
}
