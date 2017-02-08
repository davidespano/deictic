<!--
    Document   : converitoreGesti.xsl
    Created on : 23 gennaio 2017, 17.35
    Author     : root
    Description:
        Purpose of transformation follows.
-->

<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">
    <xsl:output method="text"/>
    <xsl:strip-space elements="*"/>
    <xsl:template match="/Gesture/Point">
        <xsl:value-of select="@X"/>
        <xsl:text>,-</xsl:text>
        <xsl:value-of select="@Y"/>
        <xsl:text>,1,</xsl:text>
        <xsl:value-of select="@T"/>
        <xsl:text>&#10;</xsl:text>
    </xsl:template>
</xsl:stylesheet>

