# Harmony Library Design System

A comprehensive design system for the Harmony Library book management application, featuring modern colors, typography, and reusable components.

## 🎨 Color Palette

### Primary Colors (Literary Blues)
- **Primary 50**: `#f0f4f8` - Lightest blue for backgrounds
- **Primary 500**: `#627d98` - Main brand color
- **Primary 600**: `#486581` - Primary buttons and links
- **Primary 700**: `#334e68` - Dark text and headers
- **Primary 900**: `#102a43` - Darkest blue for emphasis

### Secondary Colors (Warm Accent)
- **Secondary 500**: `#f59e0b` - Accent color for highlights
- **Secondary 600**: `#d97706` - Hover states and CTAs

### Neutral Colors
- **Neutral 50**: `#f8fafc` - Light backgrounds
- **Neutral 100**: `#f1f5f9` - Card backgrounds
- **Neutral 500**: `#64748b` - Secondary text
- **Neutral 900**: `#0f172a` - Primary text

### Semantic Colors
- **Success**: `#22c55e` - Success states
- **Warning**: `#f59e0b` - Warning states  
- **Error**: `#ef4444` - Error states

## 📝 Typography

### Font Families
- **Primary**: `Inter` - Clean, modern sans-serif for body text
- **Heading**: `Playfair Display` - Elegant serif for headings and brand
- **Mono**: `JetBrains Mono` - Code and technical content

### Font Sizes
- **xs**: `0.75rem` (12px)
- **sm**: `0.875rem` (14px)
- **base**: `1rem` (16px)
- **lg**: `1.125rem` (18px)
- **xl**: `1.25rem` (20px)
- **2xl**: `1.5rem` (24px)
- **3xl**: `1.875rem` (30px)
- **4xl**: `2.25rem` (36px)

### Font Weights
- **Light**: 300
- **Normal**: 400
- **Medium**: 500
- **Semibold**: 600
- **Bold**: 700
- **Extrabold**: 800

## 🏗️ Layout & Spacing

### Spacing Scale
- **1**: `0.25rem` (4px)
- **2**: `0.5rem` (8px)
- **3**: `0.75rem` (12px)
- **4**: `1rem` (16px)
- **5**: `1.25rem` (20px)
- **6**: `1.5rem` (24px)
- **8**: `2rem` (32px)
- **10**: `2.5rem` (40px)
- **12**: `3rem` (48px)

### Border Radius
- **sm**: `0.125rem` (2px)
- **base**: `0.25rem` (4px)
- **md**: `0.375rem` (6px)
- **lg**: `0.5rem` (8px)
- **xl**: `0.75rem` (12px)
- **2xl**: `1rem` (16px)
- **full**: `9999px` (circle)

### Shadows
- **sm**: Subtle shadow for cards
- **base**: Standard shadow for elevated elements
- **md**: Medium shadow for dropdowns
- **lg**: Large shadow for modals
- **xl**: Extra large shadow for overlays

## 🧩 Components

### Buttons

```css
/* Primary button */
.btn.btn-primary

/* Secondary button */
.btn.btn-secondary

/* Outline button */
.btn.btn-outline

/* Ghost button */
.btn.btn-ghost

/* Button sizes */
.btn.btn-sm
.btn.btn-lg
.btn.btn-xl
```

### Cards

```css
/* Basic card */
.card

/* Card sections */
.card-header
.card-body
.card-footer

/* Card content */
.card-title
.card-subtitle
```

### Forms

```css
/* Form elements */
.form-group
.form-label
.form-input
.form-textarea
.form-select

/* Form states */
.form-error
.form-help
```

### Badges

```css
/* Badge variants */
.badge.badge-primary
.badge.badge-secondary
.badge.badge-success
.badge.badge-warning
.badge.badge-error
```

### Alerts

```css
/* Alert variants */
.alert.alert-success
.alert.alert-warning
.alert.alert-error
.alert.alert-info
```

## 🎯 Usage Guidelines

### Importing Styles

```css
/* Import design system in your CSS files */
@import './styles/design-system.css';

/* Import component library */
@import './styles/components.css';
```

```javascript
// Import in React components
import '../styles/design-system.css';
```

### Using CSS Variables

```css
/* Use design system variables */
.my-component {
  color: var(--text-primary);
  background-color: var(--bg-primary);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-base);
  transition: var(--transition-base);
}
```

### Responsive Design

The design system includes responsive breakpoints:

- **sm**: `640px`
- **md**: `768px`
- **lg**: `1024px`
- **xl**: `1280px`
- **2xl**: `1536px`

### Utility Classes

```css
/* Flexbox utilities */
.flex .items-center .justify-between

/* Spacing utilities */
.m-4 .p-6 .mb-8

/* Grid utilities */
.grid .grid-cols-3 .gap-6

/* Container utilities */
.container .container-lg
```

## 🎨 Header Design Features

The redesigned header includes:

- **Sticky positioning** with backdrop blur
- **Gradient backgrounds** for modern appeal
- **Smooth animations** and hover effects
- **Responsive design** for all screen sizes
- **Professional typography** with proper hierarchy
- **Consistent spacing** using design tokens
- **Accessible focus states** and color contrast

## 📱 Responsive Behavior

- **Desktop (1024px+)**: Full navigation with all elements visible
- **Tablet (768px-1023px)**: Condensed navigation with smaller buttons
- **Mobile (767px and below)**: Stacked layout with search bar below
- **Small mobile (480px and below)**: Minimal layout with essential elements only

## 🚀 Getting Started

1. Import the design system CSS in your main CSS file
2. Use CSS variables for consistent styling
3. Apply component classes for common UI elements
4. Follow the spacing and typography guidelines
5. Test responsive behavior across breakpoints

## 🔧 Customization

To customize the design system:

1. Modify CSS variables in `design-system.css`
2. Add new component styles in `components.css`
3. Update the documentation when making changes
4. Test changes across all components

---

*This design system ensures consistency, maintainability, and a professional appearance across the entire Harmony Library application.* 