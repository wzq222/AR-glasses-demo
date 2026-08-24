# Add project specific ProGuard rules here.
-keep class com.xy.ksdk.** { *; }
-keep class com.xy.bt.** { *; }
-keep class com.xy.ota.** { *; }
-keep class org.slf4j.** { *; }
-keep class io.netty.** { *; }
-keep class com.google.protobuf.** { *; }
-dontwarn io.netty.**
-dontwarn com.google.protobuf.**
-dontwarn org.slf4j.**
